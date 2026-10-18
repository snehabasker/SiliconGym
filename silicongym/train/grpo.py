"""GRPO training on Verilog tasks with simulator rewards (TRL GRPOTrainer, 4-bit base + LoRA).

python -m silicongym.train.grpo --config silicongym/train/configs/phase1_grpo.yaml --seed 0
python -m silicongym.train.grpo --config ... --no-wandb --max-steps 5      # smoke run

The reward function runs each completion through VerilogEnv (single-turn), so training sees the
same scoring and hack flags as evaluation. Every scored sample is appended to samples.jsonl in the
run directory; config.json records the exact settings next to the results.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from silicongym.data.task import load_tasks
from silicongym.eval.evaluate import score_sample
from silicongym.rewards.verifiable import RewardConfig

ROOT = Path(__file__).resolve().parents[2]


def completion_text(c) -> str:
    return c[0]["content"] if isinstance(c, list) else c


def make_reward_fn(tasks_by_id: dict, cfg: RewardConfig, samples_path: Path, workers: int = 4):
    """TRL reward function: scores every completion in the batch in parallel and logs each sample."""
    state = {"calls": 0}
    pool = ThreadPoolExecutor(workers)

    def reward_fn(prompts, completions, task_id, **kwargs):
        ids = kwargs.get("completion_ids")
        lengths = [len(i) for i in ids] if ids is not None else [None] * len(completions)
        texts = [completion_text(c) for c in completions]
        scored = list(pool.map(lambda a: score_sample(tasks_by_id[a[0]], a[1], cfg, a[2]),
                               zip(task_id, texts, lengths)))
        state["calls"] += 1
        with samples_path.open("a") as f:
            for s in scored:
                f.write(json.dumps({"call": state["calls"], "time": time.time(), **s}) + "\n")
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"env/hack_rate": sum(s["hacked"] for s in scored) / len(scored),
                           "env/all_passed": sum(s["all_passed"] for s in scored) / len(scored),
                           "env/compiled": sum(s["compiled"] for s in scored) / len(scored)})
        except ImportError:
            pass
        return [s["reward"] for s in scored]

    return reward_fn


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ""


def filter_kwargs(cls, kwargs: dict) -> dict:
    """Drop config keys this TRL version does not know, loudly, instead of crashing or ignoring."""
    known = {f.name for f in dataclasses.fields(cls)}
    dropped = sorted(set(kwargs) - known)
    if dropped:
        print(f"[grpo] WARNING: {cls.__name__} in this TRL version has no {dropped}; ignoring them")
    return {k: v for k, v in kwargs.items() if k in known}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="run dir (default runs/<run_name>_seed<seed>)")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args(argv)

    if "WORLD_SIZE" not in os.environ:   # plain launch on a 2xT4 box: TRL rejects nn.DataParallel, so use one GPU
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    conf = yaml.safe_load(args.config.read_text())
    if args.max_steps is not None:
        conf["grpo"]["max_steps"] = args.max_steps
    run_name = conf.get("run_name", args.config.stem)
    out = args.out or ROOT / "runs" / f"{run_name}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    import torch
    import trl
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import set_seed
    from trl import GRPOConfig, GRPOTrainer

    from silicongym.train.model import LORA_TARGETS, format_prompt, load_causal_lm

    set_seed(args.seed)
    mconf, lconf = conf["model"], conf["lora"]
    tok, model = load_causal_lm(mconf["base"], mconf.get("init_adapters", []), mconf.get("load_4bit", True))
    model.train()

    tasks = load_tasks(ROOT / conf["data"]["train"])
    if conf["data"].get("max_tasks"):
        tasks = tasks[: conf["data"]["max_tasks"]]
    ds = Dataset.from_list([{"prompt": format_prompt(tok, t.spec), "task_id": t.id} for t in tasks])

    if conf["reward"].get("source", "verifiable") != "verifiable":
        raise SystemExit("train/grpo.py optimises the simulator reward only; use train/grpo_mt.py for RM/judge rewards")
    reward_cfg = RewardConfig(**{k: v for k, v in conf["reward"].items() if k in RewardConfig.__dataclass_fields__})
    rank = int(os.environ.get("RANK", 0))   # DDP: one samples file per rank, no interleaved writes
    reward_fn = make_reward_fn({t.id: t for t in tasks}, reward_cfg,
                               out / ("samples.jsonl" if rank == 0 else f"samples.rank{rank}.jsonl"),
                               conf.get("reward_workers", 4))

    use_wandb = conf.get("wandb", {}).get("enabled", True) and not args.no_wandb
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", conf.get("wandb", {}).get("project", "silicongym"))
    gpu = torch.cuda.is_available()   # T4 defaults; config values override, CPU falls back to plain fp32 AdamW
    gconf = {"fp16": gpu, "bf16": False, "gradient_checkpointing": gpu, "optim": "paged_adamw_8bit" if gpu else "adamw_torch",
             **conf["grpo"], "output_dir": str(out), "seed": args.seed,
             "report_to": "wandb" if use_wandb else "none", "run_name": f"{run_name}_seed{args.seed}"}
    trainer = GRPOTrainer(
        model=model, reward_funcs=reward_fn, train_dataset=ds, processing_class=tok,
        args=GRPOConfig(**filter_kwargs(GRPOConfig, gconf)),
        peft_config=LoraConfig(r=lconf["r"], lora_alpha=lconf["alpha"], lora_dropout=lconf["dropout"],
                               target_modules=lconf.get("targets", LORA_TARGETS), task_type="CAUSAL_LM"),
    )
    for param in trainer.model.parameters():   # TRL casts LoRA weights of 4-bit models to bf16; fp16 AMP (T4) needs fp32
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.float()
    (out / "config.json").write_text(json.dumps({
        "yaml": conf, "seed": args.seed, "args": {"config": str(args.config), "out": str(out),
                                                   "no_wandb": args.no_wandb, "max_steps": args.max_steps},
        "n_train_tasks": len(tasks), "git_commit": git_commit(), "python": platform.python_version(),
        "torch": torch.__version__, "trl": trl.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, indent=2, default=str))
    trainer.train(resume_from_checkpoint=any(out.glob("checkpoint-*")) or None)
    trainer.save_model(str(out / "final"))
    print(f"saved adapter to {out / 'final'}")


if __name__ == "__main__":
    main()
