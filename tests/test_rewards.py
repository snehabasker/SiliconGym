import pytest

from silicongym.rewards.verifiable import RewardConfig, verifiable_reward
from silicongym.tools.simulate import SimResult

CFG = RewardConfig()


def sim(passed: int, total: int, compiled=True, stdout="") -> SimResult:
    tests = {f"t{i}": i < passed for i in range(total)}
    return SimResult(compiled, tests, all_passed=passed == total, stdout=stdout)


def test_compile_failure_is_zero():
    assert verifiable_reward(sim(0, 4, compiled=False), 100, CFG)[0] == 0.0
    assert verifiable_reward(None, 100, CFG)[0] == 0.0


def test_compiles_but_fails_everything():
    assert verifiable_reward(sim(0, 4), 100, CFG)[0] == pytest.approx(0.1)


def test_partial_credit():
    assert verifiable_reward(sim(2, 4), 100, CFG)[0] == pytest.approx(0.1 + 0.8 * 0.5)


def test_all_pass_bonus_gives_one():
    assert verifiable_reward(sim(4, 4), 100, CFG)[0] == pytest.approx(1.0)


def test_length_penalty_only_beyond_threshold_and_capped():
    base = verifiable_reward(sim(4, 4), CFG.length_threshold, CFG)[0]
    longer = verifiable_reward(sim(4, 4), CFG.length_threshold + 100, CFG)[0]
    assert longer == pytest.approx(base - 100 * CFG.length_penalty)
    assert verifiable_reward(sim(4, 4), 10**6, CFG)[0] == pytest.approx(1.0 - CFG.length_penalty_cap)


def test_weights_are_configurable():
    cfg = RewardConfig(compile_bonus=0.0, partial_weight=1.0, all_pass_bonus=0.0)
    assert verifiable_reward(sim(1, 4), 10, cfg)[0] == pytest.approx(0.25)


def test_naive_ignores_per_test_and_length():
    naive = RewardConfig(variant="naive")
    assert verifiable_reward(sim(3, 4), 10**6, naive)[0] == pytest.approx(0.1)
    solved = sim(4, 4, stdout="ALL_TESTS_PASSED\n")
    assert verifiable_reward(solved, 10**6, naive)[0] == pytest.approx(1.0)


def test_naive_is_fooled_by_a_printed_token_shaped_is_not():
    spoof = sim(0, 4, stdout="ALL_TESTS_PASSED\n")
    assert verifiable_reward(spoof, 10, RewardConfig(variant="naive"))[0] == pytest.approx(1.0)
    assert verifiable_reward(spoof, 10, CFG)[0] == pytest.approx(0.1)


def test_yaml_roundtrip(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("reward:\n  partial_weight: 0.5\n  length_threshold: 99\n")
    cfg = RewardConfig.from_yaml(p)
    assert cfg.partial_weight == 0.5 and cfg.length_threshold == 99 and cfg.compile_bonus == 0.1
