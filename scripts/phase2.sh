#!/usr/bin/env bash
# Phase 2: reward-hacking study on a T4. Run after scripts/phase1_kaggle.sh succeeded.
# Trains with the naive reward, harvests flagged samples, retrains with the hardened reward, and compares both
# under the SAME strict scorer so a hacked sample cannot count as a pass for either model.
set -euo pipefail
SEEDS=${SEEDS:-"0 1 2"}
STRICT="--reward shaped --safeguards signed_output ban_sim_control multi_seed hack_penalty"
EVAL="python -m silicongym.eval.evaluate --split test --n-samples 8 $STRICT"

for s in $SEEDS; do
  python -m silicongym.train.grpo --config silicongym/train/configs/phase2_naive.yaml --seed "$s" ${NO_WANDB:+--no-wandb}
  python -m silicongym.train.grpo --config silicongym/train/configs/phase2_fixed.yaml --seed "$s" ${NO_WANDB:+--no-wandb}
  $EVAL --name naive_grpo --seed "$s" --adapter "runs/phase2_naive_seed$s/final"
  $EVAL --name fixed_grpo --seed "$s" --adapter "runs/phase2_fixed_seed$s/final"
  $EVAL --name base       --seed "$s" --kl-base none        # same strict scorer for the base model
done

# examples of each hack type found during training and evaluation + the labelled trajectory dataset
python -m silicongym.analysis.extract_hacks --samples runs/phase2_*_seed*/samples.jsonl runs/eval/*.samples.jsonl \
  --examples-out runs/hack_examples.json --dataset-out runs/trajectories.jsonl
python -m silicongym.eval.tables runs/eval --split test
