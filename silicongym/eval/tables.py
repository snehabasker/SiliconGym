"""Markdown results table from eval JSONs (mean +- std over seeds). Never hand-edit the numbers.

python -m silicongym.eval.tables runs/eval --split test
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

COLUMNS = [("pass@1", "pass@1"), ("pass@8", "pass@8"), ("compile_rate", "compile"),
           ("mean_reward", "reward"), ("hack_rate", "hack rate"), ("kl_to_base_per_token", "KL/token")]


def metric(m: dict, key: str):
    return m["hack"]["hack_rate"] if key == "hack_rate" else m.get(key)


def table(eval_dir: Path, split: str, name_filter=None) -> str:
    """name_filter: optional predicate on the run name (e.g. ablations: lambda n: "__" in n)."""
    runs: dict[str, list[dict]] = {}
    for p in sorted(Path(eval_dir).glob(f"*_{split}_seed*.json")):
        r = json.loads(p.read_text())
        if name_filter and not name_filter(r["name"]):
            continue
        runs.setdefault(r["name"], []).append(r["metrics"])
    lines = ["| model | seeds | " + " | ".join(c[1] for c in COLUMNS) + " |",
             "|---|---|" + "---|" * len(COLUMNS)]
    for name, ms in runs.items():
        cells = []
        for key, _ in COLUMNS:
            vals = [v for m in ms if (v := metric(m, key)) is not None]
            if not vals:
                cells.append("n/a")
            elif len(vals) == 1:
                cells.append(f"{vals[0]:.3f}")
            else:
                cells.append(f"{statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}")
        lines.append(f"| {name} | {len(ms)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", type=Path)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    print(table(a.eval_dir, a.split))
