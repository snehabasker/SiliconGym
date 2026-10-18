"""Build the train/val/test task set.

python -m silicongym.data.build_tasks [--out silicongym/data/tasks] [--no-validate]

1. Generate every family's tasks and assign the split of its family (families.SPLITS).
2. Validate each task against the simulator: the reference must pass every test, and an empty
   module with the same header must pass none of them.
3. Cross-split dedup using SiliconLM's shingle/near-dup code on digit-normalised spec+reference.
   Width variants normalise to the same text, so this is what proves no design leaks across splits;
   a clash drops the train/val side and keeps the test task.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from silicongym.compat import near_dup, shingles
from silicongym.tools.simulate import simulate

from .families import FAMILIES, SPLITS
from .task import Task, save_tasks

PRIORITY = {"test": 0, "val": 1, "train": 2}  # lower number survives a clash
DEFAULT_OUT = Path(__file__).parent / "tasks"


def normalise(task: Task) -> str:
    text = re.sub(r"\b\w*?(\d+)\w*\b", " N ", task.spec + " " + task.reference)
    return re.sub(r"\s+", " ", text.replace(task.module_name, "MOD")).lower()


def validate(task: Task) -> str | None:
    """Return a problem description, or None if the task behaves as a task should."""
    ref = simulate(task.reference, task.testbench, task.test_names)
    if not ref.all_passed:
        return f"reference fails: {ref.failed_tests or ref.error[:200]}"
    stub = simulate(task.module_header + "\nendmodule\n", task.testbench, task.test_names)
    if stub.pass_rate > 0 or stub.all_passed:
        return f"empty module scores {stub.pass_rate:.2f}"
    return None


def cross_split_clashes(tasks: list[Task]) -> list[tuple[str, str]]:
    """(dropped_id, kept_id) for every cross-split near-duplicate pair."""
    sh = {t.id: shingles(normalise(t)) for t in tasks}
    drops: dict[str, str] = {}
    for a in range(len(tasks)):
        for b in range(a + 1, len(tasks)):
            ta, tb = tasks[a], tasks[b]
            if ta.split != tb.split and near_dup(sh[ta.id], sh[tb.id]):
                lose, keep = (ta, tb) if PRIORITY[ta.split] > PRIORITY[tb.split] else (tb, ta)
                drops[lose.id] = keep.id
    return sorted(drops.items())


def build(out: Path, do_validate: bool = True) -> dict:
    tasks: list[Task] = []
    for split, families in SPLITS.items():
        for fam in families:
            for t in FAMILIES[fam]():
                t.split = split
                tasks.append(t)
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids"

    invalid = {}
    if do_validate:
        invalid = {t.id: p for t in tasks if (p := validate(t))}
        tasks = [t for t in tasks if t.id not in invalid]
    clashes = cross_split_clashes(tasks)
    dropped = {d for d, _ in clashes}
    tasks = [t for t in tasks if t.id not in dropped]

    out.mkdir(parents=True, exist_ok=True)
    stats = {"validated": do_validate, "invalid": invalid, "cross_split_dropped": dict(clashes)}
    for split in SPLITS:
        part = [t for t in tasks if t.split == split]
        save_tasks(part, out / f"{split}.jsonl")
        stats[split] = {"tasks": len(part), "families": sorted({t.family for t in part})}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-validate", action="store_true", help="skip simulator validation")
    args = ap.parse_args()
    print(json.dumps(build(args.out, not args.no_validate), indent=2))


if __name__ == "__main__":
    main()
