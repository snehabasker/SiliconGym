import json

import pytest
import yaml

from silicongym.analysis.extract_hacks import hack_examples, load_samples, trajectory_rows
from silicongym.data.families import FAMILIES
from silicongym.eval.agreement import agreement_report, auroc, build_rows, simulator_scorer, spearman
from silicongym.eval.regression import evaluate_humaneval, run_problem, truncate
from silicongym.rewards.ai_judge import Judge, parse_score
from silicongym.rewards.learned_rm import make_pairs
from silicongym.rewards.source import (JudgeSource, RewardModelSource, VerifiableSource,
                                       build_reward_source)
from silicongym.train.ablate import variants
from silicongym.train.rollout import group_advantages

from conftest import needs_iverilog


def a_task(family="satadd", width=4):
    return next(t for t in FAMILIES[family]() if t.params.get("width") == width)


# ---- agreement statistics ----
def test_auroc_known_values():
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [True, True, False, False]) == 0.5
    assert auroc([1, 2], [True, True]) is None


def test_spearman_known_values():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None


def test_agreement_report_finds_false_accepts_and_rejects():
    rows = [
        {"task_id": "a", "code": "ok1", "gt_pass": True, "gt_rate": 1.0, "scores": {"rm": 0.9}},
        {"task_id": "a", "code": "ok2", "gt_pass": True, "gt_rate": 1.0, "scores": {"rm": 0.2}},   # false reject
        {"task_id": "a", "code": "bad1", "gt_pass": False, "gt_rate": 0.0, "scores": {"rm": 0.8}},  # false accept
        {"task_id": "a", "code": "bad2", "gt_pass": False, "gt_rate": 0.0, "scores": {"rm": 0.1}},
    ]
    r = agreement_report(rows, {"rm": 0.5})["scorers"]["rm"]
    assert r["accuracy"] == 0.5 and r["false_accept_rate"] == 0.5 and r["false_reject_rate"] == 0.5
    assert r["false_accepts"][0]["code"] == "bad1" and r["false_rejects"][0]["code"] == "ok2"
    assert r["auroc"] == 0.75   # 3 of 4 (pass, fail) pairs ranked correctly


@needs_iverilog
def test_naive_reward_disagrees_with_ground_truth_on_a_spoofing_sample():
    t = a_task()
    spoof = (t.module_header + "\n  initial begin "
             + " ".join(f'$display("TEST {n} PASS");' for n in t.test_names)
             + ' $display("ALL_TESTS_PASSED"); end\nendmodule')
    samples = [{"task_id": t.id, "code": spoof}, {"task_id": t.id, "code": t.reference}]
    scorers = {"naive": simulator_scorer("naive", []), "shaped": simulator_scorer("shaped", [])}
    rows = build_rows(samples, {t.id: t}, scorers)
    spoof_row = rows[0]
    assert not spoof_row["gt_pass"] and spoof_row["scores"]["naive"] == pytest.approx(1.0)
    assert spoof_row["scores"]["shaped"] < 0.2 and rows[1]["gt_pass"]
    rep = agreement_report(rows, {"naive": 0.95, "shaped": 0.95})["scorers"]
    assert rep["naive"]["false_accept_rate"] == 1.0 and rep["shaped"]["false_accept_rate"] == 0.0


# ---- judge / reward sources ----
def test_judge_parses_scores_caches_and_survives_errors():
    assert parse_score("Score: 7") == 0.7 and parse_score("10") == 1.0 and parse_score("no idea") is None
    calls = []

    def complete(prompt):
        calls.append(prompt)
        if "boom" in prompt:
            raise ConnectionError("rate limited")
        return "8"
    j = Judge(complete)
    assert j.score("spec", "code") == 0.8 and j.score("spec", "code") == 0.8 and len(calls) == 1
    assert j.score("spec", "boom") is None  # error becomes None, not an exception


def test_reward_sources_zero_for_no_submission_and_count_judge_failures():
    class FakeRM:
        def score(self, spec, code):
            return 0.0
    task = a_task()
    assert VerifiableSource().score(task, {}, 0.7) == 0.7
    rm = RewardModelSource(FakeRM())
    assert rm.score(task, {"submitted": False}, 0.0) == 0.0
    assert rm.score(task, {"submitted": True, "code": "x"}, 0.0) == pytest.approx(0.5)
    js = JudgeSource(Judge(lambda p: "unparseable"))
    assert js.score(task, {"submitted": True, "code": "x"}, 0.0) == 0.0 and js.failures == 1
    with pytest.raises(ValueError):
        build_reward_source({"source": "magic"})
    assert build_reward_source({}).name == "verifiable"


# ---- learned RM data ----
def test_make_pairs_pairs_passing_with_failing_code_per_task():
    t = a_task()
    samples = [{"task_id": t.id, "code": "good1", "all_passed": True},
               {"task_id": t.id, "code": "good1", "all_passed": True},
               {"task_id": t.id, "code": "bad1", "all_passed": False},
               {"task_id": t.id, "code": "bad2", "all_passed": False},
               {"task_id": "other", "code": "x", "all_passed": True}]
    pairs = make_pairs(samples, {t.id: t})
    assert {(p["chosen"], p["rejected"]) for p in pairs} == {("good1", "bad1"), ("good1", "bad2")}
    assert all(p["prompt"] == t.spec for p in pairs)


# ---- rollout math ----
def test_group_advantages_normalise_per_group_and_zero_flat_groups():
    adv = group_advantages([1.0, 0.0, 0.5, 0.5, 0.5, 0.5], group_size=4)
    assert adv[4:] == [0.0, 0.0]                      # second group is incomplete -> still its own group
    first = group_advantages([1.0, 0.0, 1.0, 0.0], 4)
    assert sum(first) == pytest.approx(0.0) and first[0] > 0 > first[1]
    assert group_advantages([0.3] * 4, 4) == [0.0] * 4


def test_gae_and_ppo_losses():
    torch = pytest.importorskip("torch")
    from silicongym.train.ppo import gae, ppo_losses
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.zeros(1, 3)
    mask = torch.ones(1, 3)
    adv, ret = gae(rewards, values, mask, gamma=1.0, lam=1.0)
    assert adv.tolist() == [[1.0, 1.0, 1.0]] and ret.tolist() == [[1.0, 1.0, 1.0]]
    lp = torch.zeros(1, 3)
    pg, vf = ppo_losses(lp, lp, values, values, adv, ret, mask, clip=0.2, vf_coef=0.5)
    assert float(pg) == pytest.approx(-1.0) and float(vf) == pytest.approx(0.5)
    # a ratio far above 1+clip with positive advantage is clipped: gradient vanishes
    big = torch.full((1, 3), 2.0, requires_grad=True)
    pg2, _ = ppo_losses(big, lp, values, values, adv, ret, mask, 0.2, 0.5)
    pg2.backward()
    assert float(pg2) == pytest.approx(-1.2) and float(big.grad.abs().sum()) == 0.0


# ---- analysis ----
def test_extract_hacks_examples_and_trajectory_rows(tmp_path):
    run = tmp_path / "runA"
    run.mkdir()
    rows = [{"task_id": "t1", "family": "f", "reward": 1.0, "code": "evil", "pass_rate": 0.0,
             "hack_flags": [{"type": "sim_control", "detail": "uses $finish"}]},
            {"task_id": "t2", "family": "f", "reward": 0.3, "code": "fine", "hack_flags": [],
             "completions": ["a", "b"]}]
    (run / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    samples = load_samples([run / "samples.jsonl"])
    ex = hack_examples(samples)
    assert ex["sim_control"][0]["code"] == "evil" and ex["sim_control"][0]["run"] == "runA"
    assert ex["padding"] == []
    traj = trajectory_rows(samples)
    assert traj[0]["hacked"] and traj[0]["hack_types"] == ["sim_control"]
    assert not traj[1]["hacked"] and traj[1]["completions"] == ["a", "b"]


# ---- ablations ----
def test_ablation_variants_change_one_thing_and_stay_valid():
    trl_base = yaml.safe_load(open("silicongym/train/configs/phase1_grpo.yaml"))
    v = variants(trl_base, include_turns=False)
    assert v["group8"]["grpo"]["num_generations"] == 8
    g = v["group8"]["grpo"]
    assert g["per_device_train_batch_size"] % g["num_generations"] == 0
    assert g["per_device_train_batch_size"] * g["gradient_accumulation_steps"] == 16
    assert v["beta0.0"]["grpo"]["beta"] == 0.0 and trl_base["grpo"]["beta"] == 0.04  # base untouched
    from silicongym.rewards.verifiable import RewardConfig
    for c in v.values():
        RewardConfig(**c["reward"])
    mt = yaml.safe_load(open("silicongym/train/configs/phase3_multiturn.yaml"))
    mv = variants(mt, include_turns=True)
    assert mv["turns1"]["env"]["max_turns"] == 1 and mv["group2"]["grpo"]["prompts_per_step"] == 8


def test_all_shipped_configs_parse_and_have_valid_rewards():
    from pathlib import Path
    from silicongym.rewards.verifiable import RewardConfig
    fields = RewardConfig.__dataclass_fields__
    for p in Path("silicongym/train/configs").glob("phase*.yaml"):
        conf = yaml.safe_load(p.read_text())
        RewardConfig(**{k: v for k, v in conf["reward"].items() if k in fields})
        assert conf["lora"]["r"] == 16 and conf["model"]["load_4bit"] is True, p.name


# ---- capability regression ----
HE = {"task_id": "HumanEval/x", "entry_point": "double",
      "prompt": "def double(x):\n    \"\"\"double x\"\"\"\n",
      "test": "def check(f):\n    assert f(2) == 4\n    assert f(0) == 0\n"}


def test_truncate_cuts_trailing_toplevel_code():
    assert truncate("    return 2 * x\n\nprint(double(2))\n") == "    return 2 * x\n"


def test_run_problem_pass_fail_and_timeout():
    assert run_problem(HE, "    return 2 * x\n")
    assert not run_problem(HE, "    return x\n")
    assert not run_problem(HE, "    while True: pass\n", timeout=1)


def test_evaluate_humaneval_with_mock_model():
    res = evaluate_humaneval(lambda prompts: ["    return x + x\n"] * len(prompts), [HE])
    assert res["pass@1"] == 1.0 and res["n_problems"] == 1
