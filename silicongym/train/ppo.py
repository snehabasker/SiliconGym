"""PPO (phase 3) with a value head and per-token KL penalty, for comparison against GRPO.

python -m silicongym.train.ppo --config silicongym/train/configs/phase3_ppo_rm.yaml --seed 0

Differences from grpo_mt.py: a learned critic (linear head on the last hidden state), GAE advantages,
a clipped surrogate with several epochs per rollout, and the KL to the reference (LoRA disabled) folded
into the reward per token. The episode reward comes from `reward.source` (typically learned_rm) and is
paid on the last token of every turn of the episode.
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
from silicongym.train.grpo_mt import episode_record, log_metrics
from silicongym.train.rollout import completion_forward, prepare_policy, run_episodes, sample_tasks

ROOT = Path(__file__).resolve().parents[2]


def gae(rewards, values, mask, gamma: float = 1.0, lam: float = 0.95):
    """GAE over right-padded completions. rewards/values/mask: [B,T] tensors. Returns (advantages, returns)."""
    import torch
    adv = torch.zeros_like(rewards)
    last = torch.zeros(rewards.shape[0], device=rewards.device)
    for t in reversed(range(rewards.shape[1])):
        nxt = values[:, t + 1] * mask[:, t + 1] if t + 1 < rewards.shape[1] else torch.zeros_like(last)
        delta = rewards[:, t] + gamma * nxt - values[:, t]
        last = (delta + gamma * lam * last) * mask[:, t]
        adv[:, t] = last
    return adv, adv + values


def ppo_losses(logp, old_logp, values, old_values, adv, returns, mask, clip: float, vf_coef: float):
    """Clipped surrogate + clipped value loss, token-averaged over valid positions."""
    import torch
    ratio = torch.exp(logp - old_logp)
    pg = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1 - clip, 1 + clip))
    v_clipped = old_values + torch.clamp(values - old_values, -clip, clip)
    vf = torch.max((values - returns) ** 2, (v_clipped - returns) ** 2)
    n = mask.sum().clamp(min=1)
    return (pg * mask).sum() / n, vf_coef * (vf * mask).sum() / n


def pad_to(t, width):
    import torch.nn.functional as F
    return F.pad(t, (0, width - t.shape[1]))


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
    g = conf["ppo"]
    if args.max_steps is not None:
        g["max_steps"] = args.max_steps
    run_name = conf.get("run_name", args.config.stem)
    out = args.out or ROOT / "runs" / f"{run_name}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    rng = random.Random(args.seed)

    tok, model = prepare_policy(conf["model"]["base"], conf["model"].get("init_adapters", []),
                                conf["model"].get("load_4bit", True), conf["lora"])
    value_head = torch.nn.Linear(model.config.hidden_size, 1).to(model.device)
    torch.nn.init.zeros_(value_head.weight); torch.nn.init.zeros_(value_head.bias)
    tasks = load_tasks(ROOT / conf["data"]["train"])
    if conf["data"].get("max_tasks"):
        tasks = tasks[: conf["data"]["max_tasks"]]
    env_cfg = RewardConfig(**{k: v for k, v in conf["reward"].items() if k in RewardConfig.__dataclass_fields__})
    source = build_reward_source(conf["reward"])
    max_turns = conf["env"]["max_turns"]
    env_factory = lambda: VerilogEnv(env_cfg, max_turns=max_turns, timeout=conf["env"].get("timeout", 10))  # noqa: E731

    use_wandb = conf.get("wandb", {}).get("enabled", True) and not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project=conf.get("wandb", {}).get("project", "silicongym"),
                   name=f"{run_name}_seed{args.seed}", config=conf)
    policy_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": policy_params, "lr": g["learning_rate"]},
                             {"params": value_head.parameters(), "lr": g["value_lr"]}])
    (out / "config.json").write_text(json.dumps({
        "yaml": conf, "seed": args.seed, "python": platform.python_version(), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, indent=2))

    beta, clip, micro = g["kl_coef"], g["clip"], g.get("micro_batch", 4)
    for step in range(1, g["max_steps"] + 1):
        t0 = time.time()
        eps = run_episodes(model, tok, sample_tasks(tasks, g["prompts_per_step"], rng), 1, env_factory, source,
                           g["max_completion_length"], g["temperature"])
        eps = [e for e in eps if e.turns]
        flat = [(turn, e.reward) for e in eps for turn in e.turns]
        turns = [t for t, _ in flat]
        # ---- old-policy statistics (no grad) ----
        model.train()
        old, ref = [], []
        with torch.no_grad():
            for i in range(0, len(turns), micro):
                chunk = turns[i:i + micro]
                lp, mk, vl = completion_forward(model, chunk, value_head)
                with model.disable_adapter():
                    rf, _, _ = completion_forward(model, chunk)
                old.append((lp, mk, vl, rf))
        width = max(o[0].shape[1] for o in old)
        old_lp, mask, old_v, ref_lp = (torch.cat([pad_to(o[k], width) for o in old]) for k in (0, 1, 2, 3))
        rewards = -beta * (old_lp - ref_lp) * mask
        for i, (t, r) in enumerate(flat):                      # task reward lands on the last token
            rewards[i, len(t.comp_ids) - 1] += r
        adv, returns = gae(rewards, old_v, mask, g.get("gamma", 1.0), g.get("lam", 0.95))
        valid = adv[mask.bool()]
        adv = (adv - valid.mean()) / (valid.std() + 1e-8) * mask if valid.numel() > 1 else adv
        # ---- PPO epochs ----
        stats = {"pg": 0.0, "vf": 0.0, "n": 0}
        for _ in range(g["ppo_epochs"]):
            order = list(range(len(turns)))
            rng.shuffle(order)
            for i in range(0, len(order), micro):
                idx = order[i:i + micro]
                lp, mk, vl = completion_forward(model, [turns[j] for j in idx], value_head)
                w = lp.shape[1]
                pg, vf = ppo_losses(lp, old_lp[idx, :w], vl, old_v[idx, :w], adv[idx, :w], returns[idx, :w],
                                    mk, clip, g["vf_coef"])
                opt.zero_grad()
                (pg + vf).backward()
                torch.nn.utils.clip_grad_norm_(policy_params + list(value_head.parameters()), g.get("max_grad_norm", 1.0))
                opt.step()
                stats["pg"] += float(pg.detach()); stats["vf"] += float(vf.detach()); stats["n"] += 1
        kl = float(((old_lp - ref_lp) * mask).sum() / mask.sum())
        with (out / "samples.jsonl").open("a") as f:
            f.write("".join(json.dumps(episode_record(step, e)) + "\n" for e in eps))
        log_metrics(step, eps, {"kl": kl, "pg_loss": stats["pg"] / max(stats["n"], 1),
                                "vf_loss": stats["vf"] / max(stats["n"], 1), "sec": time.time() - t0},
                    out / "metrics.jsonl", use_wandb)
        if step % g.get("save_steps", 50) == 0:
            model.save_pretrained(out / f"checkpoint-{step}")
    model.save_pretrained(out / "final")
    torch.save(value_head.state_dict(), out / "value_head.pt")
    print(f"saved adapter to {out / 'final'}")


if __name__ == "__main__":
    main()
