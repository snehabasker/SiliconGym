"""Reward sources for the custom RL loops: simulator (verifiable), learned reward model, AI judge.

Every source sees the finished episode. The environment always runs the simulator, so ground-truth
fields (all_passed, hack flags) are logged whichever source drives the gradient.
"""

from __future__ import annotations

import math


class VerifiableSource:
    name = "verifiable"

    def score(self, task, info: dict, env_reward: float) -> float:
        return env_reward


class RewardModelSource:
    """Learned reward model: sigmoid(RM logit) in [0, 1]; 0 for episodes that never submitted."""
    name = "learned_rm"

    def __init__(self, reward_model):
        self.rm = reward_model

    def score(self, task, info: dict, env_reward: float) -> float:
        if not info.get("submitted") or not info.get("code"):
            return 0.0
        return 1 / (1 + math.exp(-self.rm.score(task.spec, info["code"])))


class JudgeSource:
    """LLM judge score in [0, 1]; 0 for no submission or an unparsable judgement (counted in .failures)."""
    name = "ai_judge"

    def __init__(self, judge):
        self.judge, self.failures = judge, 0

    def score(self, task, info: dict, env_reward: float) -> float:
        if not info.get("submitted") or not info.get("code"):
            return 0.0
        s = self.judge.score(task.spec, info["code"])
        if s is None:
            self.failures += 1
            return 0.0
        return s


def build_reward_source(conf: dict):
    """conf = the `reward` section of a YAML config. `source` picks the signal (default: verifiable)."""
    kind = conf.get("source", "verifiable")
    if kind == "verifiable":
        return VerifiableSource()
    if kind == "learned_rm":
        from .learned_rm import RewardModel
        return RewardModelSource(RewardModel.load(conf["rm_path"], conf.get("rm_base"), conf.get("load_4bit", True)))
    if kind == "ai_judge":
        from .ai_judge import Judge
        return JudgeSource(Judge.from_config(conf.get("judge", {})))
    raise ValueError(f"unknown reward source {kind!r}")
