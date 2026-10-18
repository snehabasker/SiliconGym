"""Generic episode interface shared by every SiliconGym domain."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Env(ABC):
    """reset(task) -> obs;  step(action) -> (obs, reward, done, info).

    `task` is a domain-specific record, `obs` and `action` are plain text. Reward is 0 until the
    episode ends. `info` always carries at least {"turn": int} and, on the final step, whatever
    the domain wants logged per sample (code, per-test results, hack flags, reward breakdown).
    """

    @abstractmethod
    def reset(self, task) -> str: ...

    @abstractmethod
    def step(self, action: str, n_tokens: int | None = None) -> tuple[str, float, bool, dict]:
        """`n_tokens`: the model's real completion length, if the caller knows it."""
