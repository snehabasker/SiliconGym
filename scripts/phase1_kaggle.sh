#!/usr/bin/env bash
# Phase 1 on a Kaggle/Colab T4 (Linux). Run cells/blocks in order; every step is resumable.
#   export HF_TOKEN=...  WANDB_API_KEY=...   (or pass --no-wandb)
#   export SFT_ADAPTERS="user/siliconlm-dapt user/siliconlm-sft"; export DPO_ADAPTERS="$SFT_ADAPTERS user/siliconlm-dpo"
set -euo pipefail
SEEDS=${SEEDS:-"0 1 2"}

apt-get install -y -qq iverilog verilator          # verilator is optional (lint tool degrades gracefully)
pip install -q -e ".[train]" pytest
git submodule update --init
python -m pytest -q                                 # all tests must pass before spending GPU time
python -m silicongym.data.build_tasks               # validates every task against the simulator
python scripts/verify_sandbox.py || true         # records what the sandbox really enforces on this machine

EVAL="python -m silicongym.eval.evaluate --split test --n-samples 8"

for s in $SEEDS; do
  # baselines: base, SiliconLM SFT, SiliconLM DPO (skipped if no adapter ids are provided)
  bash scripts/timed.sh "eval_base_s$s" $EVAL --name base --seed "$s" --kl-base none
  [ -n "${SFT_ADAPTERS:-}" ] && bash scripts/timed.sh "eval_sft_s$s" $EVAL --name sft --seed "$s" --init-adapters $SFT_ADAPTERS
  [ -n "${DPO_ADAPTERS:-}" ] && bash scripts/timed.sh "eval_dpo_s$s" $EVAL --name dpo --seed "$s" --init-adapters $DPO_ADAPTERS

  # GRPO (from the base model; edit model.init_adapters in the YAML to start from a SiliconLM checkpoint)
  bash scripts/timed.sh "train_grpo_s$s" python -m silicongym.train.grpo --config silicongym/train/configs/phase1_grpo.yaml --seed "$s" ${NO_WANDB:+--no-wandb}
  bash scripts/timed.sh "eval_grpo_s$s" $EVAL --name grpo --seed "$s" --adapter "runs/phase1_grpo_seed$s/final"
done

python -m silicongym.eval.tables runs/eval --split test
