"""Turn SiliconGym evaluation samples into a manifest of designs that passed ALL hidden tests (for eda_run.py).

python collect_generated.py --samples runs/eval/base_test_seed0.samples.jsonl ... --tag base --out manifests/generated_base.jsonl

Keeps a sample only if all_passed is true, it compiled, and no reward-hacking detector fired. Identical code (after
whitespace normalisation) for one task is synthesised once and carries `count` = how many samples produced it, so the
analysis can weight designs by how often the model actually generated them. Also writes <out>.stats.json with the
selection counts (how many samples were generated / passed / unique per task).
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

TASKS = Path(__file__).resolve().parents[1] / "silicongym" / "data" / "tasks"


def norm(code: str) -> str:
    code = re.sub(r"//[^\n]*", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    return re.sub(r"\s+", " ", code).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tasks = {}
    for split in ("train", "val", "test"):
        for line in (TASKS / f"{split}.jsonl").read_text().splitlines():
            t = json.loads(line); tasks[t["id"]] = {**t, "split": split}
    total, passed = defaultdict(int), defaultdict(int)
    uniq: dict[tuple, dict] = {}
    for path in args.samples:
        for line in path.read_text().splitlines():
            s = json.loads(line)
            total[s["task_id"]] += 1
            if not (s.get("all_passed") and s.get("compiled") and not s.get("hacked") and s.get("code")):
                continue
            passed[s["task_id"]] += 1
            key = (s["task_id"], norm(s["code"]))
            if key in uniq:
                uniq[key]["count"] += 1
            else:
                uniq[key] = {"task_id": s["task_id"], "code": s["code"], "count": 1}
    rows, per_task_k = [], defaultdict(int)
    for (tid, _), u in sorted(uniq.items()):
        t = tasks[tid]
        k = per_task_k[tid]; per_task_k[tid] += 1
        rows.append({"design_id": f"gen_{args.tag}__{tid}__{k}", "task_id": tid, "family": t["family"], "split": t["split"],
                     "kind": "generated", "top": t["module_name"], "is_seq": "clk" in t["module_header"], "clk_port": "clk",
                     "code": u["code"], "ref_code": t["reference"], "count": u["count"]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    stats = {"tag": args.tag, "samples_files": [str(p) for p in args.samples], "tasks_with_samples": len(total),
             "samples_total": sum(total.values()), "samples_passed_strict": sum(passed.values()),
             "tasks_with_a_passing_sample": len(passed), "unique_passing_designs": len(rows),
             "per_task": {t: {"samples": total[t], "passed": passed[t]} for t in sorted(total)}}
    Path(str(args.out) + ".stats.json").write_text(json.dumps(stats, indent=1))
    print({k: v for k, v in stats.items() if k != "per_task"})


if __name__ == "__main__":
    main()
