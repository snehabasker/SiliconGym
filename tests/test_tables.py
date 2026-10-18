import json

from silicongym.eval.tables import table


def write(d, name, seed, p1, hack):
    m = {"pass@1": p1, "compile_rate": 0.9, "mean_reward": 0.5, "hack": {"hack_rate": hack}}
    (d / f"{name}_test_seed{seed}.json").write_text(json.dumps({"name": name, "metrics": m}))


def test_table_means_std_and_missing_values(tmp_path):
    for seed, p1 in enumerate([0.2, 0.4]):
        write(tmp_path, "grpo", seed, p1, 0.0)
    write(tmp_path, "base", 0, 0.1, 0.05)
    out = table(tmp_path, "test")
    assert "| grpo | 2 | 0.300 ± 0.141 |" in out
    assert "| base | 1 | 0.100 | n/a |" in out and "n/a" in out
