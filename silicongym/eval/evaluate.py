"""Evaluate a model on a task split: pass@1, pass@k, compile rate, reward, hack rate, KL to base.

python -m silicongym.eval.evaluate --name grpo --adapter runs/grpo/final --split test --seed 0
python -m silicongym.eval.evaluate --name base --split test --seed 0 --kl-base none
python -m silicongym.eval.evaluate --name smoke --generator mock:reference   # no GPU, no model

Every sample goes through single-turn VerilogEnv, so scoring and hack flags are exactly what
training saw. Results land in <out>/<name>_seed<seed>.json with the full config next to the numbers.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from silicongym.compat import pass_at_k
from silicongym.data.task import Task, load_tasks
from silicongym.envs.verilog_env import VerilogEnv
from silicongym.rewards.hack_detectors import HackFlag, hack_rate
from silicongym.rewards.verifiable import RewardConfig

TASK_DIR = Path(__file__).resolve().parents[1] / "data" / "tasks"


def score_sample(task: Task, completion: str, cfg: RewardConfig, n_tokens: int | None = None) -> dict:
    env = VerilogEnv(cfg, max_turns=1)
    env.reset(task)
    _, reward, _, info = env.step(completion, n_tokens)
    return {"task_id": task.id, "family": task.family, "reward": reward, "completion": completion, **info}


def aggregate(samples: list[dict], ks: tuple[int, ...] = (1, 4, 8)) -> dict:
    """Metrics from per-sample dicts (needs task_id, all_passed, compiled, pass_rate, reward, hack_flags)."""
    by_task: dict[str, list[dict]] = {}
    for s in samples:
        by_task.setdefault(s["task_id"], []).append(s)
    n = min(len(v) for v in by_task.values())
    out = {"n_tasks": len(by_task), "samples_per_task": n, "n_samples": len(samples)}
    for k in ks:
        if k <= n:
            out[f"pass@{k}"] = sum(
                pass_at_k(len(v), sum(x["all_passed"] for x in v), k) for v in by_task.values()
            ) / len(by_task)
    out["compile_rate"] = sum(s["compiled"] for s in samples) / len(samples)
    out["mean_test_pass_rate"] = sum(s["pass_rate"] for s in samples) / len(samples)
    out["mean_reward"] = sum(s["reward"] for s in samples) / len(samples)
    out["hack"] = hack_rate([[HackFlag(**f) for f in s["hack_flags"]] for s in samples])
    return out


# ---- generators: (tasks, n) -> list[list[(completion, n_tokens)]] -----------------------------

def mock_generator(kind: str):
    def gen(tasks: list[Task], n: int):
        text = {"reference": lambda t: t.reference,
                "empty": lambda t: t.module_header + "\nendmodule"}[kind]
        return [[(text(t), None)] * n for t in tasks]
    return gen


def hf_generator(args):
    import torch
    from silicongym.train.model import format_prompt, load_causal_lm

    tok, model = load_causal_lm(args.base, args.init_adapters, not args.no_4bit, args.adapter)
    samples_logp: dict = {}

    def gen(tasks: list[Task], n: int):
        out = []
        for t in tasks:
            ids = tok(format_prompt(tok, t.spec), return_tensors="pt").to(model.device)
            with torch.no_grad():
                seqs = model.generate(
                    **ids, do_sample=True, temperature=args.temperature, top_p=0.95,
                    num_return_sequences=n, max_new_tokens=args.max_new_tokens,
                    pad_token_id=tok.pad_token_id)
            plen = ids["input_ids"].shape[1]
            rows = []
            for s in seqs:
                comp = s[plen:]
                eos = (comp == tok.eos_token_id).nonzero()
                comp = comp[: int(eos[0]) + 1] if len(eos) else comp
                rows.append((tok.decode(comp, skip_special_tokens=True), len(comp)))
                samples_logp.setdefault(t.id, []).append((ids["input_ids"][0].cpu(), comp.cpu()))
            out.append(rows)
        return out

    gen.model, gen.tok, gen.sequences = model, tok, samples_logp
    return gen


def kl_to_base(gen, args) -> dict:
    """Mean per-token log-ratio log pi_model - log pi_base over the model's own samples (k1 estimator)."""
    import gc
    import torch
    from silicongym.train.model import load_causal_lm

    def seq_logps(model, prompt, comp):
        full = torch.cat([prompt, comp]).unsqueeze(0).to(model.device)
        with torch.no_grad():
            logits = model(full).logits[0, len(prompt) - 1:-1].float()
        return torch.log_softmax(logits, -1).gather(1, comp.to(model.device).unsqueeze(1)).squeeze(1)

    pairs = [(p, c) for rows in gen.sequences.values() for p, c in rows if len(c)]
    own = [seq_logps(gen.model, p, c).cpu() for p, c in pairs]
    del gen.model
    gc.collect(); torch.cuda.empty_cache()
    _, base = load_causal_lm(args.base, [], not args.no_4bit)
    ref = [seq_logps(base, p, c).cpu() for p, c in pairs]
    diffs = torch.cat([a - b for a, b in zip(own, ref)])
    return {"kl_to_base_per_token": float(diffs.mean()), "kl_tokens": int(diffs.numel())}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--tasks", type=Path, default=None, help="override tasks jsonl")
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--init-adapters", nargs="*", default=[], help="adapters merged in order (SiliconLM chain)")
    ap.add_argument("--adapter", default=None, help="final LoRA adapter (e.g. GRPO output)")
    ap.add_argument("--generator", default="hf", help="hf | mock:reference | mock:empty")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--reward", choices=["shaped", "naive"], default="shaped")
    ap.add_argument("--safeguards", nargs="*", default=[],
                    help="signed_output ban_sim_control multi_seed hack_penalty; score with all four for ground truth")
    ap.add_argument("--kl-base", choices=["compute", "none"], default="compute")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/eval"))
    args = ap.parse_args(argv)

    random.seed(args.seed)
    tasks = load_tasks(args.tasks or TASK_DIR / f"{args.split}.jsonl")[: args.max_tasks]
    cfg = RewardConfig(variant=args.reward, safeguards=args.safeguards)

    if args.generator.startswith("mock:"):
        gen, kl = mock_generator(args.generator.split(":", 1)[1]), {}
    else:
        import torch
        from transformers import set_seed
        set_seed(args.seed)
        gen = hf_generator(args)
    completions = gen(tasks, args.n_samples)

    jobs = [(t, c, nt) for t, rows in zip(tasks, completions) for c, nt in rows]
    with ThreadPoolExecutor(args.workers) as pool:
        samples = list(pool.map(lambda j: score_sample(j[0], j[1], cfg, j[2]), jobs))
    metrics = aggregate(samples)
    if not args.generator.startswith("mock:"):
        metrics.update(kl_to_base(gen, args) if args.kl_base == "compute" and (args.adapter or args.init_adapters)
                       else {"kl_to_base_per_token": 0.0 if args.kl_base == "compute" else None})

    result = {"name": args.name, "config": {**vars(args), "out": str(args.out), "tasks": str(args.tasks),
                                            "python": platform.python_version(), "argv": sys.argv},
              "reward_config": cfg.to_dict(), "metrics": metrics}
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.name}_{args.split}_seed{args.seed}"
    (args.out / f"{stem}.json").write_text(json.dumps(result, indent=2))
    (args.out / f"{stem}.samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
