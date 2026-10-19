import json

from silicongym.analysis import report as R


def eval_json(d, name, seed, p1, hack, by_type):
    m = {"pass@1": p1, "compile_rate": 0.9, "mean_reward": 0.4,
         "hack": {"hack_rate": hack, "by_type": by_type}}
    (d / f"{name}_test_seed{seed}.json").write_text(json.dumps({"name": name, "metrics": m}))


def test_every_section_degrades_to_a_no_data_message(tmp_path):
    for out in (R.eval_table(tmp_path), R.hack_by_type(tmp_path), R.hack_examples_md(tmp_path / "x.json"),
                R.agreement_md(tmp_path / "x.json"), R.regression_md(tmp_path)):
        assert "no data found" in out
    assert R.load_curves(tmp_path) == {}


def test_hack_by_type_and_filtered_table(tmp_path):
    types = {"trivial_module": 0, "sim_control": 6, "hardcoded_outputs": 0, "fs_access": 0, "padding": 0}
    eval_json(tmp_path, "naive_grpo", 0, 0.3, 0.25, types)
    eval_json(tmp_path, "fixed_grpo", 0, 0.3, 0.0, {k: 0 for k in types})
    md = R.hack_by_type(tmp_path)
    assert "| naive_grpo | 1 | 0.250 | 0.0 | 6.0 |" in md and "| fixed_grpo | 1 | 0.000 |" in md
    only_fixed = R.eval_table(tmp_path, "test", lambda n: n.startswith("fixed"))
    assert "fixed_grpo" in only_fixed and "naive_grpo" not in only_fixed


def test_curves_from_both_log_formats(tmp_path):
    a = tmp_path / "custom_seed0"; a.mkdir()
    (a / "metrics.jsonl").write_text(json.dumps({"step": 1, "reward": 0.2, "all_passed": 0.0, "hack_rate": 0.0}) + "\n")
    b = tmp_path / "trl_seed0"; b.mkdir()
    rows = [{"call": 1, "reward": 0.1, "all_passed": False, "hacked": False},
            {"call": 1, "reward": 0.9, "all_passed": True, "hacked": True}]
    (b / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    c = R.load_curves(tmp_path)
    assert c["custom_seed0"][0]["reward"] == 0.2
    assert c["trl_seed0"] == [{"step": 1, "reward": 0.5, "all_passed": 0.5, "hack_rate": 0.5}]


def test_agreement_and_regression_tables(tmp_path):
    ex = {"task_id": "t", "score": 0.9, "gt_rate": 0.0, "code": "module x; endmodule"}
    rep = {"n": 10, "n_ground_truth_pass": 4, "scorers": {"learned_rm": {
        "auroc": 0.8, "spearman_vs_pass_rate": None, "accuracy": 0.7, "false_accept_rate": 0.3,
        "false_reject_rate": 0.1, "false_accepts": [ex], "false_rejects": []}}}
    (tmp_path / "agreement.json").write_text(json.dumps(rep))
    md = R.agreement_md(tmp_path / "agreement.json")
    assert "| learned_rm | 0.800 | n/a | 0.700 | 0.300 | 0.100 |" in md and "false accept" in md
    (tmp_path / "r.json").write_text(json.dumps({"name": "g", "model": {"pass@1": 0.5}, "base": {"pass@1": 0.55},
                                                 "delta_pass@1": -0.05}))
    assert "| g | 0.500 | 0.550 | -0.050 |" in R.regression_md(tmp_path)


def test_gpu_hours_table_and_total(tmp_path):
    p = tmp_path / "t.jsonl"
    rows = [{"stage": "train_grpo_s0", "seconds": 3600, "gpu_hours": 1.0, "gpu": "Tesla T4", "exit_code": 0},
            {"stage": "eval_s0", "seconds": 1800, "gpu_hours": 0.5, "gpu": None, "exit_code": 0}]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    md = R.gpu_hours_md(p)
    assert "| train_grpo_s0 | 60.0 | 1.000 | Tesla T4 | 0 |" in md and "| **total** | 90.0 | 1.500 |" in md
    assert "no data found" in R.gpu_hours_md(tmp_path / "missing.jsonl")
