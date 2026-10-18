"""Generate ablation configs from a base YAML (phase 4).

python -m silicongym.train.ablate --base silicongym/train/configs/phase1_grpo.yaml --out silicongym/train/configs/ablations
python -m silicongym.train.ablate --base silicongym/train/configs/phase3_multiturn.yaml --turns --out ...

Each config changes exactly one thing relative to the base and is named for it. Group size keeps the
completions per optimizer step constant (16) so memory and data per step stay comparable.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def variants(base: dict, include_turns: bool) -> dict[str, dict]:
    out: dict[str, dict] = {}

    def make(name: str, edit) -> None:
        c = copy.deepcopy(base)
        edit(c)
        c["run_name"] = f"{base.get('run_name', 'run')}__{name}"
        out[name] = c

    for g in (2, 8):
        def group(c, g=g):
            c["grpo"]["num_generations"] = g
            if "per_device_train_batch_size" in c["grpo"]:       # TRL path
                c["grpo"]["per_device_train_batch_size"] = g
                c["grpo"]["gradient_accumulation_steps"] = 16 // g
            else:                                                 # custom loop: 16 episodes per step
                c["grpo"]["prompts_per_step"] = 16 // g
        make(f"group{g}", group)
    for beta in (0.0, 0.2):
        make(f"beta{beta}", lambda c, b=beta: c["grpo"].__setitem__("beta", b))
    make("reward_binary", lambda c: c["reward"].update(partial_weight=0.0, all_pass_bonus=0.9))
    make("reward_no_compile_bonus", lambda c: c["reward"].update(compile_bonus=0.0, partial_weight=0.9))
    make("reward_partial_heavy", lambda c: c["reward"].update(compile_bonus=0.0, partial_weight=1.0, all_pass_bonus=0.0))
    if include_turns:                                             # needs the custom multi-turn loop
        make("turns1", lambda c: c["env"].__setitem__("max_turns", 1))
        make("turns4", lambda c: c["env"].__setitem__("max_turns", 4))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--turns", action="store_true", help="add single- vs multi-turn variants")
    a = ap.parse_args()
    base = yaml.safe_load(a.base.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    for name, conf in variants(base, a.turns).items():
        path = a.out / f"{base.get('run_name', a.base.stem)}__{name}.yaml"
        path.write_text(yaml.safe_dump(conf, sort_keys=False))
        print(path)


if __name__ == "__main__":
    main()
