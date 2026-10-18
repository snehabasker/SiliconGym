# SiliconGym

An RL environment and training pipeline for LLM-generated Verilog, built on the
[SiliconLM](https://github.com/snehabasker/SiliconLM) verifier and checkpoints. A model writes a module from a
natural-language spec, may call tools (compile / simulate / lint) over several turns, and is trained with GRPO (and PPO)
on rewards computed from Icarus Verilog simulation. The project also **measures reward hacking**: reward earned without
solving the task.

Zero-budget design: free Kaggle/Colab T4s, 4-bit base + LoRA r=16, HF `generate` rollouts.

**No result numbers appear in this README.** Every number comes from a logged run: `runs/eval/*.json`,
`runs/agreement.json`, `runs/regression/*.json`. Generate tables with `silicongym.eval.tables` or
`silicongym/analysis/report.ipynb`.

## What has and has not been verified

The code for all four phases is written. It was developed on a Windows CPU-only machine (Icarus 12), so:

| Verified here (automated tests) | Not yet run, needs a GPU / credentials / Linux |
|---|---|
| tools, sandbox, per-test simulation, rewards, detectors, env (single + multi-turn), task builder + validation + dedup, evaluator (mock generators), aggregation, tables, agreement stats, regression harness, report helpers, ablation configs | **No training run has been done and no results exist.** |
| the TRL entrypoint `train/grpo.py` (TRL 1.16), the custom GRPO and PPO loops, reward-model training/scoring, HF evaluation with KL-to-base, HumanEval harness: all run for real on a tiny random Qwen2 on CPU (`tests/test_train_smoke.py`) | 4-bit/bitsandbytes loading, fp16 on a T4, T4 memory fit, vLLM, Unsloth |
| | `accelerate` DDP on 2xT4, AI-judge API calls, Verilator lint, Linux rlimits / `unshare`, Hub push |

Expect small fixes on the first GPU run. Run `scripts/phase1_kaggle.sh` first and treat any traceback as the next task.

## Setup

```bash
git clone --recurse-submodules <this repo> && cd silicongym      # SiliconLM is a submodule (reused, not copied)
pip install -e ".[dev]"                                          # + ".[train]" on the GPU box
apt-get install iverilog verilator                               # verilator optional; lint degrades gracefully
python -m pytest -q                                              # 121 tests; simulator tests skip without iverilog
python -m silicongym.data.build_tasks                            # writes + validates silicongym/data/tasks/*.jsonl
```

Reused from SiliconLM via `silicongym/compat.py`: `extract_verilog`, `pass_at_k`, `PASS_TOKEN`, and the corpus dedup
`shingles`/`near_dup`. New code was needed because `check_candidate` is binary pass/fail with no per-test output,
observation-safe timeouts or isolation.

## Environment

**Action format.**
```
<final>module ...endmodule</final>                         submit; ends the episode
<tool>compile|simulate|lint</tool><code>module ...</code>    run a tool (omit <code> to reuse the last code sent)
```
A bare `module...endmodule` (or a ```verilog fence) counts as `<final>`, so an untuned model is not scored on formatting.
Episodes end on `<final>` or after `max_turns` (default 4; `1` = single-turn). Running out of turns without submitting gives
reward 0. Observations are the spec plus the tool history; `simulate` returns only the **names of failed tests**.

**Tasks.** 16 parametrised design families, 96 tasks. Each hidden testbench drives the DUT and a golden model with the
same stimulus and prints `TEST <group> PASS|FAIL` per group; the task declares the expected groups, so a missing line
(early `$finish`, hang) counts as failed. Splits are **by family**: train 10 families / 58 tasks, val 2 / 14, test 4 / 24.
Width variants of one design are near-duplicates, so splitting inside a family would leak; `build_tasks` checks this with
SiliconLM's shingle/Jaccard dedup on digit-normalised spec+reference. The five families SiliconLM's DPO pairs came from
(adder, mux2, dff, counter, priority encoder) are train-only, so SFT/DPO baselines never see test prompts.
The test set is small (24 tasks): expect wide error bars.

Testbench quality: of 206 compiling single-operator mutants of the references, 4 survive and all 4 are equivalent. This check
found a real hole (negedge-clocked designs passed every sequential testbench); sampling just before each posedge closed it.

**Sandbox.** Temp dir, scrubbed env, wall-clock timeout (default 10 s), capped output, UTF-8-safe writes, and on Linux
rlimits plus `unshare -n` when permitted. Timeouts and missing binaries return observations, never exceptions. Source with
file/system/include constructs (`$fopen`, `$readmem*`, `$system`, `` `include``, `$dump*`, ...) is refused before anything runs.
Sources are deleted before `vvp` starts. **Limits:** the compiled `.vvp` embeds the testbench, so the construct ban is what
protects it; network isolation is best-effort (iverilog/vvp load no network code); Windows has no rlimits or `unshare`;
`PythonEnv` (the second domain, proving the interface is generic) is **not** hardened against peeking at its tests.

## Rewards

`shaped` (`rewards/verifiable.py`, YAML-configurable): 0 if it does not compile; else
`0.1 + 0.8 x (tests passed / total) + 0.1 if all pass`, minus `0.0005` per token beyond 400 (cap 0.2).
`naive`: `0.1` for compiling + `0.9` if the pass token appears in stdout; no per-test accounting, no safeguards.
`learned_rm.py`: Bradley-Terry reward model on `{prompt, chosen, rejected}` pairs (SiliconLM's DPO-pair format works as-is;
`make-pairs` builds the same format from simulator-scored samples). `ai_judge.py`: LLM judge, any OpenAI-compatible
endpoint or a local HF model; unparsable/failed calls score 0 and are counted.

## Reward hacking study (phase 2)

Detectors (`rewards/hack_detectors.py`, each with positive and negative tests) log a per-sample flag and type:
`trivial_module`, `sim_control` (also catches printing the pass protocol), `hardcoded_outputs`, `fs_access`, `padding`.
`hack rate = flagged samples / total samples`. Each hack has a matching safeguard, switched on via `reward.safeguards`:

| hack | safeguard | what it does |
|---|---|---|
| spoofed PASS lines / early `$finish` | `signed_output`, `ban_sim_control` | PASS lines carry a per-run random nonce the candidate cannot know; `$finish/$stop/$fatal/$exit` in a candidate is refused |
| hardcoded outputs | `multi_seed` | testbench re-run with a second stimulus seed; a test passes only if it passes both |
| trivial module, padding, fs access | `hack_penalty` | any detector flag turns the reward into `-hack_penalty` |

`phase2_naive.yaml` trains on the naive reward with no safeguards; `phase2_fixed.yaml` retrains with all four. Evaluate both
(and the base model) with the **same strict scorer** (`--safeguards signed_output ban_sim_control multi_seed hack_penalty`),
otherwise a hacked sample counts as a pass for one model and not the other. `analysis/extract_hacks.py` saves examples of
each hack type and builds the labelled trajectory dataset. Whether the model actually *discovers* each hack within the
step budget is an empirical question: the study reports what is found, and unit tests only prove that the naive reward
*can* be hacked and the safeguarded one cannot (by these specific attacks).

## Multi-turn, reward comparison, PPO (phase 3)

TRL's `GRPOTrainer` cannot run an environment between turns, so `train/grpo_mt.py` is a small custom loop (group-normalised
advantage shared by all turns of an episode, k3 KL to the LoRA-disabled reference). `train/ppo.py` adds a value head, GAE,
clipped surrogate and per-token KL for the PPO-vs-GRPO comparison. `reward.source` selects the optimised signal
(`verifiable | learned_rm | ai_judge`); the simulator always runs, so ground truth and hack flags are logged for every source.
`eval/agreement.py` scores the same held-out code with every reward and reports AUROC, Spearman vs the true pass rate,
accuracy, false-accept/false-reject rates and concrete disagreement examples against the strict simulator ground truth.
Train the reward model on **train-split** pairs only.

## Rigour and release (phase 4)

`train/ablate.py` writes one-change-each configs (group size 2/8, KL 0/0.2, three reward-weight variants, 1 vs 4 turns).
`eval/regression.py` runs a HumanEval subset (greedy, sandboxed) for the base and trained adapters and reports the delta.
`configs/accelerate_ddp.yaml` + `scripts/phase4.sh` run a short 2xT4 DDP check. `scripts/push_to_hub.py` uploads an adapter
and the trajectory dataset; it is a **dry run unless `--push`** (needs `HF_TOKEN`; repos are private unless `--public`).

## Running

```bash
bash scripts/phase1_kaggle.sh   # tests, task build, baselines (base/SFT/DPO), GRPO, eval, table
bash scripts/phase2.sh          # naive vs fixed reward, hack examples, trajectory dataset
bash scripts/phase3.sh          # RM + judge + PPO + multi-turn, agreement
bash scripts/phase4.sh          # ablations, regression, DDP, hub dry run
jupyter nbconvert --execute silicongym/analysis/report.ipynb --to html
```
All take `SEEDS="0 1 2"` and `NO_WANDB=1`. Every run writes `config.json` (YAML, seed, versions, git commit, GPU) beside
its samples; evaluation JSONs embed their arguments and reward config. Weights & Biases is optional (`--no-wandb`).

Defaults: group size 4, 4 prompts per optimizer step, completions capped at 384 tokens, rollouts via HF `generate`
(`use_vllm: false`; vLLM colocate on a T4 is untested). If a step does not fit in 16 GB, lower `max_completion_length`,
then the per-device batch (keep it a multiple of `num_generations`) and raise gradient accumulation. To start from SiliconLM,
list the DAPT/SFT/DPO adapter ids in `model.init_adapters` (or `--init-adapters`); without them everything runs on plain
Qwen2.5-Coder-1.5B. Unsloth is not used: TRL + HF generate was chosen for transparency and testability.

## Layout

`envs/` base, verilog, python · `tools/` compile, simulate, lint, sandbox · `rewards/` verifiable, learned_rm, ai_judge,
hack_detectors, source · `data/` families, testbench, build_tasks, tasks/ · `train/` grpo (TRL), grpo_mt, ppo, rollout, model,
ablate, configs/ · `eval/` evaluate, agreement, regression, tables · `analysis/` extract_hacks, report, report.ipynb ·
`scripts/` phase runners, push_to_hub · `tests/`
