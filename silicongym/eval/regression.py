"""General-coding capability check (HumanEval subset) to detect capability loss after RL on Verilog.

python -m silicongym.eval.regression --name grpo --adapter runs/phase1_grpo_seed0/final --n-problems 40 --compare-base

Greedy completion of the HumanEval prompt as plain text (no chat template: these are base-style models),
truncated at the first top-level statement after the function, executed in the same sandbox as the
Verilog tools. Reports pass@1 on the subset, and with --compare-base the drop relative to the base model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from silicongym.tools.sandbox import Workspace

STOPS = re.compile(r"\n(?=def |class |if __name__|print\(|assert |#)")


def truncate(completion: str) -> str:
    """Cut at the first new top-level construct after the function body."""
    m = STOPS.search(completion)
    return completion[: m.start()] if m else completion


def run_problem(problem: dict, completion: str, timeout: float = 10.0) -> bool:
    program = (problem["prompt"] + truncate(completion) + "\n\n" + problem["test"]
               + f"\n\ncheck({problem['entry_point']})\n")
    with Workspace() as ws:
        ws.write("prog.py", program)
        res = ws.run([sys.executable, "prog.py"], timeout)
    return res.returncode == 0 and not res.timed_out


def evaluate_humaneval(complete, problems: list[dict]) -> dict:
    """complete(prompts) -> completions. Returns pass@1 and per-problem results."""
    completions = complete([p["prompt"] for p in problems])
    results = {p["task_id"]: run_problem(p, c) for p, c in zip(problems, completions)}
    return {"n_problems": len(problems), "pass@1": sum(results.values()) / len(results), "per_problem": results}


def load_humaneval(n: int, path: Path | None = None) -> list[dict]:
    if path:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        from datasets import load_dataset
        rows = list(load_dataset("openai/openai_humaneval", split="test"))
    return rows[:n]


def hf_complete(base: str, init_adapters: list[str], adapter: str | None, load_4bit: bool, max_new_tokens: int):
    import torch
    from silicongym.train.model import load_causal_lm

    tok, model = load_causal_lm(base, init_adapters, load_4bit, adapter)

    def complete(prompts: list[str]) -> list[str]:
        out = []
        for p in prompts:
            ids = tok(p, return_tensors="pt").to(model.device)
            with torch.no_grad():
                g = model.generate(**ids, do_sample=False, max_new_tokens=max_new_tokens,
                                   pad_token_id=tok.pad_token_id)
            out.append(tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
        return out
    complete.free = lambda: [model.cpu(), torch.cuda.empty_cache()]
    return complete


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--init-adapters", nargs="*", default=[])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n-problems", type=int, default=40)
    ap.add_argument("--problems", type=Path, default=None, help="local HumanEval-format jsonl (else HF datasets)")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--compare-base", action="store_true")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/regression"))
    args = ap.parse_args(argv)

    problems = load_humaneval(args.n_problems, args.problems)
    result = {"name": args.name, "config": {k: str(v) for k, v in vars(args).items()}}
    model = hf_complete(args.base, args.init_adapters, args.adapter, not args.no_4bit, args.max_new_tokens)
    result["model"] = evaluate_humaneval(model, problems)
    if args.compare_base:
        model.free()
        base = evaluate_humaneval(hf_complete(args.base, [], None, not args.no_4bit, args.max_new_tokens), problems)
        result["base"] = base
        result["delta_pass@1"] = result["model"]["pass@1"] - base["pass@1"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.name}_seed{args.seed}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2, default=lambda o: None)[:1500])
    return result


if __name__ == "__main__":
    main()
