"""Upload a trained adapter and the labelled trajectory dataset to the Hugging Face Hub.

python scripts/push_to_hub.py --adapter runs/phase1_grpo_seed0/final --adapter-repo snehabasker/silicongym-grpo \
    --dataset runs/trajectories.jsonl --dataset-repo snehabasker/silicongym-trajectories          # dry run
python scripts/push_to_hub.py ... --push                                                         # needs HF_TOKEN

Dry run by default: it prints exactly what would be uploaded. Publishing is outward-facing and the
data may be cached or indexed even if deleted later, so --push is deliberate. Repos are created
private unless --public is given.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

CARD = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-1.5B
library_name: peft
tags: [verilog, rl, grpo, reward-hacking]
---
# {name}

LoRA adapter trained with SiliconGym ({config_hint}). Numbers for this adapter live in the SiliconGym
repository's generated results tables; none are reproduced here.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--adapter-repo")
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--dataset-repo")
    ap.add_argument("--config-hint", default="see config.json in the run directory")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args()

    plan = []
    if a.adapter:
        files = sorted(p.name for p in a.adapter.iterdir())
        plan.append(("model", a.adapter_repo, a.adapter, files))
    if a.dataset:
        n = sum(1 for _ in a.dataset.open())
        flagged = sum(json.loads(line).get("hacked", False) for line in a.dataset.open())
        plan.append(("dataset", a.dataset_repo, a.dataset, [f"{a.dataset.name} ({n} rows, {flagged} flagged)"]))
    for kind, repo, path, files in plan:
        print(f"[{'PUSH' if a.push else 'dry-run'}] {kind} {repo or '<missing --*-repo>'} <- {path}: {files}")
    if not a.push:
        return
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("set HF_TOKEN to push")
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    for kind, repo, path, _ in plan:
        if not repo:
            raise SystemExit(f"--{kind if kind == 'dataset' else 'adapter'}-repo is required with --push")
        api.create_repo(repo, repo_type=kind, private=not a.public, exist_ok=True)
        if kind == "model":
            (path / "README.md").write_text(CARD.format(name=repo.split("/")[-1], config_hint=a.config_hint))
            api.upload_folder(folder_path=str(path), repo_id=repo, repo_type="model")
        else:
            api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=repo, repo_type="dataset")
        print(f"uploaded {kind} to https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}")


if __name__ == "__main__":
    main()
