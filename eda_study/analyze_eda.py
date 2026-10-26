"""Compare synthesised model-generated designs with the golden designs of the same tasks.

python analyze_eda.py --golden results/golden/results.jsonl --generated results/generated/results.jsonl \
                      --out results/compare_generated_vs_golden

Unit of analysis is the TASK (several samples of one task are not independent): per task, the mean over that
task's functionally-correct samples (identical designs weighted by their sample count) of metric/golden_metric; then a geometric mean over tasks with a
percentile bootstrap over tasks (10,000 resamples, fixed seed). Also reports a sign test and the share of tasks
where the generated design is larger / equal / smaller. Writes summary.json, per_task.csv and summary.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

METRICS = [("cells", "cell count"), ("area_um2", "synth area (um^2)"), ("place_area_um2", "placed area (um^2)"),
           ("data_arrival_ns", "critical path delay (ns)"), ("power_total_w", "total power (W)")]


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def ok(r: dict, key: str) -> bool:
    return r.get("status") == "ok" and r.get(key) is not None and (key not in ("place_area_um2", "data_arrival_ns",
                                                                                "power_total_w") or r.get("pnr_status") == "ok")


def gmean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def boot_ci(per_task: list[float], n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    stats = sorted(gmean([rng.choice(per_task) for _ in per_task]) for _ in range(n))
    return stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def sign_test_p(pos: int, neg: int) -> float:
    """Two-sided exact binomial sign test, ties dropped."""
    n, k = pos + neg, min(pos, neg)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, required=True)
    ap.add_argument("--generated", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only-equivalent", action="store_true",
                    help="keep only generated samples proven equivalent to the golden design")
    args = ap.parse_args()

    gold = {r["task_id"]: r for r in load(args.golden)}
    gen = defaultdict(list)
    for r in load(args.generated):
        if args.only_equivalent and not r.get("equiv_to_golden"):
            continue
        gen[r["task_id"]].append(r)

    summary: dict = {"n_tasks_with_generated": len(gen), "n_generated_designs": sum(len(v) for v in gen.values()),
                     "only_equivalent": args.only_equivalent, "metrics": {}}
    rows = []
    for key, label in METRICS:
        ratios, larger, smaller, equal = [], 0, 0, 0
        for tid, rs in sorted(gen.items()):
            g = gold.get(tid)
            if not g or not ok(g, key) or not g[key]:
                continue
            good = [r for r in rs if ok(r, key) and r[key] > 0]
            if not good:
                continue
            w = [r.get("count", 1) for r in good]        # weight = how many samples produced this exact design
            rr = [r[key] / g[key] for r in good]
            m = sum(a * b for a, b in zip(rr, w)) / sum(w)
            ratios.append(m)
            larger += m > 1 + 1e-9
            smaller += m < 1 - 1e-9
            equal += abs(m - 1) <= 1e-9
            rows.append({"metric": key, "task_id": tid, "n_samples": sum(w), "golden": g[key],
                         "generated_mean": sum(r[key] * c for r, c in zip(good, w)) / sum(w), "mean_ratio": m})
        if not ratios:
            continue
        lo, hi = boot_ci(ratios)
        summary["metrics"][key] = {"label": label, "n_tasks": len(ratios), "geomean_ratio": gmean(ratios),
                                   "ci95": [lo, hi], "tasks_larger": larger, "tasks_equal": equal,
                                   "tasks_smaller": smaller, "sign_test_p": sign_test_p(larger, smaller)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.out / "per_task.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "task_id", "n_samples", "golden", "generated_mean", "mean_ratio"])
        w.writeheader(); w.writerows(rows)
    lines = [f"Generated designs: {summary['n_generated_designs']} over {summary['n_tasks_with_generated']} tasks"
             f" (equivalent-only: {args.only_equivalent})", "",
             "| metric | tasks | geomean ratio (gen/golden) | 95% CI | larger / equal / smaller | sign-test p |",
             "|---|---|---|---|---|---|"]
    for k, m in summary["metrics"].items():
        lines.append(f"| {m['label']} | {m['n_tasks']} | {m['geomean_ratio']:.3f} | [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}] |"
                     f" {m['tasks_larger']} / {m['tasks_equal']} / {m['tasks_smaller']} | {m['sign_test_p']:.3g} |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
