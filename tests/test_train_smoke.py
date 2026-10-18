"""CPU smoke tests: tiny random Qwen2 + real tokenizer, 2 optimizer steps through each training loop.

They prove the plumbing (rollout, multi-turn env stepping, log-probs, LoRA updates, checkpoints, logging),
not learning. Skipped without transformers/peft or without network access to fetch the tokenizer.
"""

import json
import random

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from conftest import needs_iverilog  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
    except Exception as e:  # offline
        pytest.skip(f"tokenizer unavailable: {e}")
    d = tmp_path_factory.mktemp("tiny")
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    torch.manual_seed(0)
    Qwen2ForCausalLM(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)


class RandomSource:
    """Stand-in reward so groups have non-zero variance and the update path is exercised."""
    name = "random"

    def __init__(self):
        self.rng = random.Random(0)

    def score(self, task, info, env_reward):
        return self.rng.random()


def write_conf(tmp_path, model_dir, section, extra):
    conf = {"run_name": "smoke", "model": {"base": model_dir, "init_adapters": [], "load_4bit": False},
            "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "targets": ["q_proj", "v_proj"]},
            "data": {"train": "silicongym/data/tasks/train.jsonl", "max_tasks": 4},
            "env": {"max_turns": 2, "timeout": 5},
            "reward": {"source": "verifiable", "variant": "shaped", "safeguards": ["ban_sim_control"]},
            "wandb": {"enabled": False}, section: extra}
    p = tmp_path / "conf.yaml"
    p.write_text(yaml.safe_dump(conf))
    return p


def run_dir_checks(out, steps):
    metrics = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert [m["step"] for m in metrics] == list(range(1, steps + 1))
    assert all(0 <= m["turns"] <= 2 and "hack_rate" in m for m in metrics)
    assert (out / "final" / "adapter_config.json").exists() and (out / "config.json").exists()
    samples = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    assert samples and "completions" in samples[0] and "hack_flags" in samples[0]
    return metrics


@needs_iverilog
def test_grpo_mt_two_steps_multi_turn(tiny_model_dir, tmp_path, monkeypatch):
    from silicongym.train import grpo_mt
    monkeypatch.setattr(grpo_mt, "build_reward_source", lambda conf: RandomSource())
    conf = write_conf(tmp_path, tiny_model_dir, "grpo", {
        "num_generations": 2, "prompts_per_step": 2, "max_completion_length": 12, "temperature": 1.0,
        "learning_rate": 1e-3, "beta": 0.04, "max_steps": 2, "micro_batch": 2, "save_steps": 1})
    out = tmp_path / "out"
    grpo_mt.main(["--config", str(conf), "--seed", "0", "--out", str(out), "--no-wandb"])
    metrics = run_dir_checks(out, 2)
    assert metrics[0]["trained_turns"] > 0           # random rewards -> non-flat groups -> a real update
    assert (out / "checkpoint-1").exists()


@needs_iverilog
def test_ppo_two_steps(tiny_model_dir, tmp_path, monkeypatch):
    from silicongym.train import ppo
    monkeypatch.setattr(ppo, "build_reward_source", lambda conf: RandomSource())
    conf = write_conf(tmp_path, tiny_model_dir, "ppo", {
        "prompts_per_step": 4, "max_completion_length": 12, "temperature": 1.0, "learning_rate": 1e-3,
        "value_lr": 1e-3, "kl_coef": 0.04, "clip": 0.2, "vf_coef": 0.1, "ppo_epochs": 2, "max_steps": 2,
        "micro_batch": 2, "save_steps": 100})
    out = tmp_path / "out"
    ppo.main(["--config", str(conf), "--seed", "0", "--out", str(out), "--no-wandb"])
    metrics = run_dir_checks(out, 2)
    assert "pg_loss" in metrics[0] and "vf_loss" in metrics[0]
    assert (out / "value_head.pt").exists()


@needs_iverilog
def test_rollout_logprobs_match_between_generation_and_forward(tiny_model_dir):
    """The log-probs the trainer recomputes must be the log-probs of the sampled tokens (greedy sanity)."""
    from silicongym.train.model import load_causal_lm
    from silicongym.train.rollout import Turn, completion_forward
    tok, model = load_causal_lm(tiny_model_dir, [], load_4bit=False)
    ids = tok("module m;", return_tensors="pt").input_ids[0].tolist()
    comp = [ids[-1], ids[0]]
    logp, mask, _ = completion_forward(model, [Turn(ids, comp, "")])
    assert logp.shape == (1, 2) and mask.sum() == 2 and bool((logp <= 0).all())
    # batching with a longer sequence (right padding) must not change a sequence's log-probs
    other = Turn(ids + ids, comp + comp, "")
    solo, _, _ = completion_forward(model, [Turn(ids, comp, "")])
    both, _, _ = completion_forward(model, [Turn(ids, comp, ""), other])
    assert torch.allclose(solo[0], both[0, :2], atol=1e-4)


# ---- evaluation / RM / regression through real (tiny) HF models ----
@pytest.fixture(scope="module")
def tiny_adapter(tiny_model_dir, tmp_path_factory):
    """A LoRA with randomised B matrices, so it differs measurably from the base model."""
    from silicongym.train.rollout import prepare_policy
    _, model = prepare_policy(tiny_model_dir, [], False, {"r": 4, "alpha": 8, "dropout": 0.0,
                                                          "targets": ["q_proj", "v_proj", "down_proj"]})
    torch.manual_seed(1)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.5)
    d = tmp_path_factory.mktemp("adapter")
    model.save_pretrained(d)
    return str(d)


@needs_iverilog
def test_evaluate_hf_path_computes_positive_kl_for_a_changed_policy(tiny_model_dir, tiny_adapter, tmp_path):
    from silicongym.eval.evaluate import main
    common = ["--base", tiny_model_dir, "--no-4bit", "--split", "test", "--max-tasks", "2", "--n-samples", "2",
              "--max-new-tokens", "8", "--out", str(tmp_path), "--seed", "0"]
    m = main(["--name", "tuned", "--adapter", tiny_adapter, *common])
    assert m["n_samples"] == 4 and m["kl_tokens"] > 0 and m["kl_to_base_per_token"] != 0.0
    base = main(["--name", "base", *common])
    assert base["kl_to_base_per_token"] == 0.0
    saved = json.loads((tmp_path / "tuned_test_seed0.json").read_text())
    assert saved["config"]["adapter"] == tiny_adapter and "pass@1" in saved["metrics"]


def test_reward_model_trains_and_scores_on_cpu(tiny_model_dir, tmp_path):
    from silicongym.rewards.learned_rm import RewardModel, train_rm
    pairs = [{"prompt": f"spec {i}", "chosen": "assign y = a + b;", "rejected": "assign y = 0;"} for i in range(12)]
    stats = train_rm(pairs, tmp_path / "rm", tiny_model_dir, seed=0, epochs=3, lr=5e-3, batch_size=4,
                     load_4bit=False)
    assert stats["n_train_pairs"] == 11 and 0.0 <= stats["val_pair_accuracy"] <= 1.0
    rm = RewardModel.load(str(tmp_path / "rm"), tiny_model_dir, load_4bit=False)
    good, bad = rm.score("spec 0", "assign y = a + b;"), rm.score("spec 0", "assign y = 0;")
    assert good > bad    # it fits the (trivially separable) training pairs


def test_regression_hf_path_runs_end_to_end(tiny_model_dir, tmp_path):
    from silicongym.eval.regression import main
    prob = {"task_id": "HumanEval/x", "entry_point": "double", "prompt": "def double(x):\n",
            "test": "def check(f):\n    assert f(2) == 4\n"}
    pf = tmp_path / "he.jsonl"
    pf.write_text(json.dumps(prob) + "\n")
    res = main(["--name", "tiny", "--base", tiny_model_dir, "--no-4bit", "--problems", str(pf), "--n-problems", "1",
                "--max-new-tokens", "6", "--compare-base", "--out", str(tmp_path / "reg")])
    assert res["model"]["n_problems"] == 1 and res["delta_pass@1"] == 0.0   # same weights, greedy decoding


@needs_iverilog
def test_trl_grpo_entrypoint_two_steps(tiny_model_dir, tmp_path):
    pytest.importorskip("trl")
    pytest.importorskip("datasets")
    if torch.cuda.device_count() > 1:
        pytest.skip("TRL rejects nn.DataParallel; grpo.py pins one GPU, but only for a fresh process")
    from silicongym.train import grpo
    conf = write_conf(tmp_path, tiny_model_dir, "grpo", {
        "num_generations": 2, "per_device_train_batch_size": 2, "gradient_accumulation_steps": 1,
        "max_completion_length": 8, "temperature": 1.0, "learning_rate": 1e-3, "beta": 0.04,
        "max_steps": 2, "logging_steps": 1, "save_steps": 100, "use_vllm": False, "report_to": "none"})
    out = tmp_path / "out"
    grpo.main(["--config", str(conf), "--seed", "0", "--out", str(out), "--no-wandb"])
    rows = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    assert len(rows) >= 4 and {"reward", "hack_flags", "task_id"} <= set(rows[0])
    assert (out / "final" / "adapter_config.json").exists()
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["seed"] == 0 and cfg["trl"] and cfg["yaml"]["grpo"]["num_generations"] == 2
