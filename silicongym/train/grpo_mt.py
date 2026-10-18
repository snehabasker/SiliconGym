"""Custom GRPO loop for multi-turn tool use and non-simulator reward sources (learned RM, AI judge).

python -m silicongym.train.grpo_mt --config silicongym/train/configs/phase3_multiturn.yaml --seed 0

TRL's GRPOTrainer (train/grpo.py) covers single-turn simulator rewards. This loop exists because TRL
cannot run an environment between turns. Update rule per optimizer step:
  1. sample `prompts_per_step` tasks, roll out `num_generations` episodes each (up to env.max_turns turns);
  2. advantage = group-normalised episode reward, shared by every turn of the episode;
  3. loss = -A * mean_t log pi(token) + beta * mean_t KL_k3(pi || pi_ref), pi_ref = LoRA disabled.
One gradient step per rollout, so the policy is exactly on-policy and no ratio clipping is needed.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import yaml

from silicongym.data.task import load_tasks
from silicongym.envs.verilog_env import VerilogEnv
from silicongym.rewards.source import build_reward_source
from silicongym.rewards.verifiable import RewardConfig
from silicongym.train.rollout import (completion_forward, group_advantages, prepare_policy,
                                      run_episodes, sample_tasks)

ROOT = Path(__file__).resolve().parents[2]


def episode_record(step: int, ep) -> dict:
    return {"step": step, "task_id": ep.task.id, "family": ep.task.family, "reward": ep.reward,
            "env_reward": ep.env_reward, "n_turns": len(ep.turns), "completions": [t.text for t in ep.turns],
            **{k: v for k, v in ep.info.items() if k not in ("kind",)}}


def log_metrics(step: int, eps, extra: dict, path: Path, use_wandb: bool) -> dict:
    n = len(eps)
    m = {"step": step, "reward": sum(e.reward for e in eps) / n,
         "env_reward": sum(e.env_reward for e in eps) / n,
         "all_passed": sum(bool(e.info.get("all_passed")) for e in eps) / n,
         "compiled": sum(bool(e.info.get("compiled")) for e in eps) / n,
         "hack_rate": sum(bool(e.info.get("hacked")) for e in eps) / n,
         "submitted": sum(bool(e.info.get("submitted")) for e in eps) / n,
         "turns": sum(len(e.turns) for e in eps) / n, **extra}
    with path.open("a") as f:
        f.write(json.dumps(m) + "\n")
    if use_wandb:
        import wandb
        wandb.log(m, step=step)
    print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()), flush=True)
    return m


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args(argv)

    import torch
    from transformers import set_seed

    conf = yaml.safe_load(args.config.read_text())
    g = conf["grpo"]
    if args.max_steps is not None:
        g["max_steps"] = args.max_steps
    run_name = conf.get("run_name", args.config.stem)
    out = args.out or ROOT / "runs" / f"{run_name}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    rng = random.Random(args.seed)

    tok, model = prepare_policy(conf["model"]["base"], conf["model"].get("init_adapters", []),
                                conf["model"].get("load_4bit", True), conf["lora"])
    tasks = load_tasks(ROOT / conf["data"]["train"])
    if conf["data"].get("max_tasks"):
        tasks = tasks[: conf["data"]["max_tasks"]]
    # the environment always runs the simulator; `source` decides which signal is optimised
    env_cfg = RewardConfig(**{k: v for k, v in conf["reward"].items()
                              if k in RewardConfig.__dataclass_fields__})
    source = build_reward_source(conf["reward"])
    max_turns = conf["env"]["max_turns"]
    env_factory = lambda: VerilogEnv(env_cfg, max_turns=max_turns, timeout=conf["env"].get("timeout", 10))  # noqa: E731

    use_wandb = conf.get("wandb", {}).get("enabled", True) and not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project=conf.get("wandb", {}).get("project", "silicongym"),
                   name=f"{run_name}_seed{args.seed}", config=conf)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=g["learning_rate"])
    (out / "config.json").write_text(json.dumps({
        "yaml": conf, "seed": args.seed, "python": platform.python_version(), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, indent=2))

    G, beta, micro = g["num_generations"], g["beta"], g.get("micro_batch", 4)
    for step in range(1, g["max_steps"] + 1):
        t0 = time.time()
        batch = sample_tasks(tasks, g["prompts_per_step"], rng)
        eps = run_episodes(model, tok, batch, G, env_factory, source, g["max_completion_length"], g["temperature"])
        adv = group_advantages([e.reward for e in eps], G)
        flat = [(turn, a) for e, a in zip(eps, adv) for turn in e.turns if a != 0.0]
        kl_acc, n_tok = 0.0, 0
        model.train()
        opt.zero_grad()
        for i in range(0, len(flat), micro):
            chunk = flat[i:i + micro]
            turns = [t for t, _ in chunk]
            a = torch.tensor([x for _, x in chunk], device=model.device).unsqueeze(1)
            with torch.no_grad(), model.disable_adapter():
                ref_logp, _, _ = completion_forward(model, turns)
            logp, mask, _ = completion_forward(model, turns)
            kl = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1
            per_seq = (-(a * logp) + beta * kl) * mask
            loss = (per_seq.sum(1) / mask.sum(1)).sum() / len(flat)
            loss.backward()
            kl_acc += float((kl * mask).sum().detach()); n_tok += int(mask.sum())
        if flat:
            torch.nn.utils.clip_grad_norm_(params, g.get("max_grad_norm", 1.0))
            opt.step()
        with (out / "samples.jsonl").open("a") as f:
            f.write("".join(json.dumps(episode_record(step, e)) + "\n" for e in eps))
        log_metrics(step, eps, {"kl": kl_acc / max(n_tok, 1), "trained_turns": len(flat),
                                "sec": time.time() - t0}, out / "metrics.jsonl", use_wandb)
        if step % g.get("save_steps", 50) == 0:
            model.save_pretrained(out / f"checkpoint-{step}")
    model.save_pretrained(out / "final")
    print(f"saved adapter to {out / 'final'}")


if __name__ == "__main__":
    main()
