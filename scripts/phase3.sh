#!/usr/bin/env bash
# Phase 3: multi-turn tool use, learned RM, AI judge, PPO, and reward-vs-ground-truth agreement.
# Needs: runs from phase 1/2 (for RM training data), GROQ_API_KEY (or edit judge config) for the judge runs.
set -euo pipefail
SEEDS=${SEEDS:-"0"}
STRICT="--reward shaped --safeguards signed_output ban_sim_control multi_seed hack_penalty"
EVAL="python -m silicongym.eval.evaluate --split test --n-samples 8 $STRICT"

# 1. reward model: preference pairs from TRAIN-split samples (passing vs failing code), or your own SiliconLM
#    dpo_pairs.jsonl in the same {prompt, chosen, rejected} format.
python -m silicongym.eval.evaluate --name rmdata --split train --n-samples 8 --seed 100 $STRICT --kl-base none
python -m silicongym.rewards.learned_rm make-pairs --samples runs/eval/rmdata_train_seed100.samples.jsonl \
  runs/phase1_grpo_seed0/samples.jsonl --out runs/rm_pairs.jsonl
python -m silicongym.rewards.learned_rm train --pairs runs/rm_pairs.jsonl --out runs/rm --seed 0

for s in $SEEDS; do
  for cfg in phase3_multiturn phase3_rm phase3_judge; do
    python -m silicongym.train.grpo_mt --config "silicongym/train/configs/$cfg.yaml" --seed "$s" ${NO_WANDB:+--no-wandb}
    $EVAL --name "$cfg" --seed "$s" --adapter "runs/${cfg}_seed$s/final"
  done
  python -m silicongym.train.ppo --config silicongym/train/configs/phase3_ppo_rm.yaml --seed "$s" ${NO_WANDB:+--no-wandb}
  $EVAL --name phase3_ppo_rm --seed "$s" --adapter "runs/phase3_ppo_rm_seed$s/final"
done

# 2. agreement of each reward with simulator ground truth on held-out (test) samples of the trained policies
python -m silicongym.eval.agreement --samples runs/eval/phase3_*_test_seed0.samples.jsonl \
  --rm runs/rm --judge-config silicongym/train/configs/phase3_judge.yaml --out runs/agreement.json
python -m silicongym.eval.tables runs/eval --split test
