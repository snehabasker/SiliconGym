"""Collect flagged samples into per-type examples and a labelled trajectory dataset.

python -m silicongym.analysis.extract_hacks --samples runs/*/samples.jsonl \
    --examples-out runs/hack_examples.json --dataset-out runs/trajectories.jsonl

Works on samples written by train/grpo.py, grpo_mt.py, ppo.py and eval/evaluate.py. The trajectory
dataset is what scripts/push_to_hub.py uploads: one row per sample with its reward, ground-truth
fields and hack labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from silicongym.rewards.hack_detectors import HACK_TYPES


def load_samples(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        run = p.parent.name if p.name == "samples.jsonl" else p.name.removesuffix(".samples.jsonl")
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append({"run": run, **json.loads(line)})
    return rows


def hack_types(sample: dict) -> list[str]:
    return sorted({f["type"] for f in sample.get("hack_flags", [])})


def hack_examples(samples: list[dict], per_type: int = 5) -> dict[str, list[dict]]:
    """Up to `per_type` examples of each hack type, preferring the ones that earned the most reward."""
    out: dict[str, list[dict]] = {t: [] for t in HACK_TYPES}
    for s in sorted(samples, key=lambda s: -s.get("reward", 0)):
        for t in hack_types(s):
            if len(out[t]) < per_type:
                detail = next(f["detail"] for f in s["hack_flags"] if f["type"] == t)
                out[t].append({"run": s["run"], "task_id": s["task_id"], "reward": s.get("reward"),
                               "pass_rate": s.get("pass_rate"), "detail": detail, "code": s.get("code", "")})
    return out


def trajectory_rows(samples: list[dict]) -> list[dict]:
    rows = []
    for s in samples:
        completions = s.get("completions") or ([s["completion"]] if "completion" in s else [])
        rows.append({
            "run": s["run"], "task_id": s["task_id"], "family": s.get("family"), "step": s.get("step"),
            "completions": completions, "code": s.get("code", ""), "reward": s.get("reward"),
            "env_reward": s.get("env_reward", s.get("reward")), "compiled": s.get("compiled"),
            "all_passed": s.get("all_passed"), "pass_rate": s.get("pass_rate"),
            "hacked": bool(s.get("hack_flags")), "hack_types": hack_types(s),
            "hack_details": [f["detail"] for f in s.get("hack_flags", [])],
        })
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", type=Path, required=True)
    ap.add_argument("--per-type", type=int, default=5)
    ap.add_argument("--examples-out", type=Path, default=Path("runs/hack_examples.json"))
    ap.add_argument("--dataset-out", type=Path, default=Path("runs/trajectories.jsonl"))
    a = ap.parse_args(argv)
    samples = load_samples(a.samples)
    ex = hack_examples(samples, a.per_type)
    a.examples_out.parent.mkdir(parents=True, exist_ok=True)
    a.examples_out.write_text(json.dumps(ex, indent=2))
    rows = trajectory_rows(samples)
    a.dataset_out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} samples, {sum(r['hacked'] for r in rows)} flagged; examples per type: "
          + ", ".join(f"{t}={len(v)}" for t, v in ex.items()))


if __name__ == "__main__":
    main()
