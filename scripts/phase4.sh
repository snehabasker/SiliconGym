#!/usr/bin/env bash
# Phase 4: ablations, capability regression, 2xT4 DDP check, and (dry-run) Hub upload plan.
set -euo pipefail
STRICT="--reward shaped --safeguards signed_output ban_sim_control multi_seed hack_penalty"
EVAL="python -m silicongym.eval.evaluate --split test --n-samples 8 $STRICT"
SEED=${SEED:-0}

# 1. ablations: group size, KL coefficient, reward weights (TRL path) and single- vs multi-turn (custom loop)
python -m silicongym.train.ablate --base silicongym/train/configs/phase2_fixed.yaml --out silicongym/train/configs/ablations
python -m silicongym.train.ablate --base silicongym/train/configs/phase3_multiturn.yaml --turns \
  --out silicongym/train/configs/ablations
for cfg in silicongym/train/configs/ablations/phase2_fixed__*.yaml; do
  name=$(basename "$cfg" .yaml)
  python -m silicongym.train.grpo --config "$cfg" --seed "$SEED" ${NO_WANDB:+--no-wandb}
  $EVAL --name "$name" --seed "$SEED" --adapter "runs/${name}_seed$SEED/final"
done
for cfg in silicongym/train/configs/ablations/phase3_multiturn__turns*.yaml; do
  name=$(basename "$cfg" .yaml)
  python -m silicongym.train.grpo_mt --config "$cfg" --seed "$SEED" ${NO_WANDB:+--no-wandb}
  $EVAL --name "$name" --seed "$SEED" --adapter "runs/${name}_seed$SEED/final"
done

# 2. capability regression: base vs each trained adapter on a HumanEval subset
python -m silicongym.eval.regression --name base --n-problems 40 --seed "$SEED"
for a in runs/phase2_fixed_seed$SEED runs/phase3_multiturn_seed$SEED; do
  python -m silicongym.eval.regression --name "$(basename "$a")" --adapter "$a/final" --n-problems 40 --compare-base --seed "$SEED"
done

# 3. multi-GPU: short DDP run on Kaggle 2xT4 (needs `Accelerator: GPU T4 x2`). Compare samples/sec and the
#    reward curve with the single-GPU run; global batch doubles (2 ranks x 4 completions x 4 accumulation).
accelerate launch --config_file silicongym/train/configs/accelerate_ddp.yaml -m silicongym.train.grpo \
  --config silicongym/train/configs/phase2_fixed.yaml --seed "$SEED" --max-steps 20 --no-wandb --out runs/ddp_check

# 4. Hub upload: DRY RUN. Review the plan, then add --push (needs HF_TOKEN) to publish.
python -m silicongym.analysis.extract_hacks --samples runs/*/samples.jsonl --examples-out runs/hack_examples.json \
  --dataset-out runs/trajectories.jsonl
python scripts/push_to_hub.py --adapter "runs/phase2_fixed_seed$SEED/final" --adapter-repo snehabasker/silicongym-grpo \
  --dataset runs/trajectories.jsonl --dataset-repo snehabasker/silicongym-trajectories
