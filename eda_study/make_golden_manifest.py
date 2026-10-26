"""Write a manifest of the golden (reference) designs of every SiliconGym task, for eda_run.py.

python make_golden_manifest.py --out manifests/golden.jsonl
"""

import argparse
import json
from pathlib import Path

TASKS = Path(__file__).resolve().parents[1] / "silicongym" / "data" / "tasks"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for split in ("train", "val", "test"):
        for line in (TASKS / f"{split}.jsonl").read_text().splitlines():
            t = json.loads(line)
            rows.append({"design_id": f"golden__{t['id']}", "task_id": t["id"], "family": t["family"],
                         "split": split, "kind": "golden", "top": t["module_name"],
                         "is_seq": "clk" in t["module_header"], "clk_port": "clk", "code": t["reference"]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} golden designs -> {args.out}")


if __name__ == "__main__":
    main()
