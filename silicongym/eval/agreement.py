"""How well does each reward signal agree with simulator ground truth on held-out samples?

python -m silicongym.eval.agreement --samples runs/eval/grpo_test_seed0.samples.jsonl \
    --rm runs/rm --judge-config silicongym/train/configs/judge.yaml --out runs/agreement.json

Ground truth = the strict simulator score (signed output, sim-control ban, multi-seed): did the code
pass every test under stimulus it could not have memorised. Each scorer rates the same code; we report
AUROC, Spearman vs the true test pass rate, and accuracy / false-accept / false-reject rates at a
per-scorer acceptance threshold, plus concrete disagreement examples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from silicongym.data.task import Task, load_tasks
from silicongym.envs.verilog_env import VerilogEnv
from silicongym.rewards.verifiable import RewardConfig

STRICT = ["signed_output", "ban_sim_control", "multi_seed", "hack_penalty"]
DEFAULT_THRESHOLDS = {"verifiable_naive": 0.95, "verifiable_shaped": 0.95, "learned_rm": 0.5, "ai_judge": 0.5}


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auroc(scores: list[float], labels: list[bool]) -> float | None:
    """P(random positive scores above random negative), ties count half. None if one class is empty."""
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    r = _ranks(scores)
    return (sum(rk for rk, y in zip(r, labels) if y) - pos * (pos + 1) / 2) / (pos * neg)


def spearman(x: list[float], y: list[float]) -> float | None:
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (vx * vy) ** 0.5


def agreement_report(rows: list[dict], thresholds: dict[str, float], examples: int = 3) -> dict:
    """rows: {task_id, code, gt_pass: bool, gt_rate: float, scores: {scorer: float}}."""
    labels = [r["gt_pass"] for r in rows]
    out: dict = {"n": len(rows), "n_ground_truth_pass": sum(labels), "scorers": {}}
    for name in sorted({k for r in rows for k in r["scores"]}):
        used = [r for r in rows if r["scores"].get(name) is not None]
        sc, lb = [r["scores"][name] for r in used], [r["gt_pass"] for r in used]
        thr = thresholds.get(name, 0.5)
        accept = [s >= thr for s in sc]
        fa = [r for r, a in zip(used, accept) if a and not r["gt_pass"]]
        fr = [r for r, a in zip(used, accept) if not a and r["gt_pass"]]
        n_fail, n_pass = lb.count(False), lb.count(True)
        out["scorers"][name] = {
            "n_scored": len(used), "threshold": thr,
            "auroc": auroc(sc, lb), "spearman_vs_pass_rate": spearman(sc, [r["gt_rate"] for r in used]),
            "accuracy": sum(a == y for a, y in zip(accept, lb)) / len(used) if used else None,
            "false_accept_rate": len(fa) / n_fail if n_fail else None,
            "false_reject_rate": len(fr) / n_pass if n_pass else None,
            "false_accepts": [_example(r, name) for r in sorted(fa, key=lambda r: -r["scores"][name])[:examples]],
            "false_rejects": [_example(r, name) for r in sorted(fr, key=lambda r: r["scores"][name])[:examples]],
        }
    return out


def _example(r: dict, name: str) -> dict:
    return {"task_id": r["task_id"], "score": r["scores"][name], "gt_rate": r["gt_rate"], "code": r["code"]}


def ground_truth(task: Task, code: str) -> tuple[bool, float]:
    env = VerilogEnv(RewardConfig(safeguards=STRICT), max_turns=1)
    env.reset(task)
    _, _, _, info = env.step(f"<final>{code}</final>")
    return bool(info["all_passed"]) and not info["hacked"], float(info["pass_rate"])


def simulator_scorer(variant: str, safeguards: list[str]):
    cfg = RewardConfig(variant=variant, safeguards=safeguards)

    def score(task: Task, code: str) -> float:
        env = VerilogEnv(cfg, max_turns=1)
        env.reset(task)
        return env.step(f"<final>{code}</final>")[1]
    return score


def build_rows(samples: list[dict], tasks: dict[str, Task], scorers: dict) -> list[dict]:
    cache: dict[tuple, dict] = {}
    rows = []
    for s in samples:
        code, tid = s.get("code") or "", s["task_id"]
        if not code or tid not in tasks:
            continue
        key = (tid, code)
        if key not in cache:
            gt_pass, gt_rate = ground_truth(tasks[tid], code)
            cache[key] = {"task_id": tid, "code": code, "gt_pass": gt_pass, "gt_rate": gt_rate,
                          "scores": {n: f(tasks[tid], code) for n, f in scorers.items()}}
        rows.append(cache[key])
    return rows


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, default=Path("silicongym/data/tasks/test.jsonl"))
    ap.add_argument("--rm", default=None, help="learned reward model adapter dir")
    ap.add_argument("--rm-base", default=None)
    ap.add_argument("--judge-config", type=Path, default=None, help="YAML with a `judge:` section")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/agreement.json"))
    args = ap.parse_args(argv)

    scorers = {"verifiable_naive": simulator_scorer("naive", []),
               "verifiable_shaped": simulator_scorer("shaped", [])}
    if args.rm:
        import math
        from silicongym.rewards.learned_rm import RewardModel
        rm = RewardModel.load(args.rm, args.rm_base)
        scorers["learned_rm"] = lambda t, c: 1 / (1 + math.exp(-rm.score(t.spec, c)))
    if args.judge_config:
        import yaml
        from silicongym.rewards.ai_judge import Judge
        judge = Judge.from_config(yaml.safe_load(args.judge_config.read_text()).get("judge", {}))
        scorers["ai_judge"] = lambda t, c: judge.score(t.spec, c)

    samples = [json.loads(line) for p in args.samples for line in p.read_text().splitlines() if line.strip()]
    rows = build_rows(samples[: args.max_samples], {t.id: t for t in load_tasks(args.tasks)}, scorers)
    report = agreement_report(rows, DEFAULT_THRESHOLDS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    for name, m in report["scorers"].items():
        print(f"{name:18s} auroc={m['auroc']} spearman={m['spearman_vs_pass_rate']} "
              f"acc={m['accuracy']} false_accept={m['false_accept_rate']} false_reject={m['false_reject_rate']}")
    return report


if __name__ == "__main__":
    main()
