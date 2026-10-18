"""Verilog task environment.

Action format (documented in README):
    <final>module ...endmodule</final>        submit and end the episode
    <tool>compile|simulate|lint</tool>        run a tool on the code in the <code> tag, e.g.
    <code>module ...endmodule</code>          (or the last code given, if <code> is omitted)
A bare module (or a ```verilog fence) with no tags counts as <final>, so an untuned model is not
scored on formatting. Single-turn mode is max_turns=1: whatever the model writes is its submission.
"""

from __future__ import annotations

import re

from silicongym.compat import extract_verilog
from silicongym.data.task import Task
from silicongym.rewards.hack_detectors import detect_hacks
from silicongym.rewards.verifiable import RewardConfig, verifiable_reward
from silicongym.tools.compile import compile_verilog
from silicongym.tools.lint import lint_verilog
from silicongym.tools.sandbox import DEFAULT_TIMEOUT_S
from silicongym.tools.simulate import simulate

from .base import Env

_TOOLS = ("compile", "simulate", "lint")
_TAG = lambda name: re.compile(rf"<{name}>(.*?)</{name}>", re.S)  # noqa: E731
_FINAL, _TOOL, _CODE = _TAG("final"), _TAG("tool"), _TAG("code")


def approx_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def parse_action(text: str) -> tuple[str, str, str]:
    """-> (kind, tool_name, code) with kind in {"final", "tool", "invalid"}."""
    if m := _FINAL.search(text):
        return "final", "", extract_verilog(m.group(1))
    if m := _TOOL.search(text):
        name = m.group(1).strip().lower()
        code = _CODE.search(text)
        return ("tool", name, extract_verilog(code.group(1)) if code else "")
    if "endmodule" in text:
        return "final", "", extract_verilog(text)
    return "invalid", "", ""


class VerilogEnv(Env):
    def __init__(self, reward_cfg: RewardConfig | None = None, max_turns: int = 4,
                 timeout: float = DEFAULT_TIMEOUT_S, detect: bool = True):
        self.cfg = reward_cfg or RewardConfig()
        self.max_turns, self.timeout, self.detect = max_turns, timeout, detect

    def reset(self, task: Task) -> str:
        self.task, self.turn, self.done = task, 0, False
        self.history: list[str] = []
        self.last_code = ""
        return self._observation()

    def _observation(self) -> str:
        head = f"Task:\n{self.task.spec}\n"
        if self.max_turns > 1:
            head += (f"\nYou have {self.max_turns - self.turn} turn(s) left. Use "
                     "<tool>compile|simulate|lint</tool><code>...</code> to test code, "
                     "or <final>...</final> to submit.\n")
        return head + "".join(f"\n[turn {i + 1}] {h}" for i, h in enumerate(self.history))

    def step(self, action: str, n_tokens: int | None = None) -> tuple[str, float, bool, dict]:
        assert not self.done, "episode finished; call reset()"
        self.turn += 1
        kind, tool, code = parse_action(action)
        if code:
            self.last_code = code
        info: dict = {"turn": self.turn, "kind": kind}

        if kind == "final":
            return self._finish(code, action, n_tokens, info)
        if kind == "tool" and tool in _TOOLS:
            self.history.append(f"{tool}: {self._run_tool(tool, code or self.last_code)}")
        else:
            self.history.append("invalid action: use <tool>compile|simulate|lint</tool> or <final>...</final>")
        if self.turn >= self.max_turns:  # out of turns without submitting
            self.done = True
            info.update(code=self.last_code, submitted=False, reward_parts={"compiled": False}, n_tokens=0,
                        compiled=False, pass_rate=0.0, all_passed=False, tests={}, timed_out=False,
                        hack_flags=[], hacked=False)
            return self._observation(), 0.0, True, info
        return self._observation(), 0.0, False, info

    def _simulate(self, code: str):
        return simulate(code, self.task.testbench, self.task.test_names, self.timeout,
                        **self.cfg.sim_kwargs)

    def _run_tool(self, tool: str, code: str) -> str:
        if not code:
            return "no code provided"
        if tool == "compile":
            res = compile_verilog(code, self.timeout)
            if res.ok:
                return "compiled OK"
            return "compile FAILED: " + ("; ".join(f"line {e['line']}: {e['msg']}" for e in res.errors[:5])
                                          or res.stderr[:300])
        if tool == "simulate":  # failed test names only: the testbench itself stays hidden
            res = self._simulate(code)
            if not res.compiled:
                return "does not compile: " + res.error[:200]
            return ("all tests passed" if res.all_passed else
                    f"{len(res.failed_tests)}/{len(res.tests)} tests failed: {', '.join(res.failed_tests)}"
                    + (" (simulation timed out)" if res.timed_out else ""))
        res = lint_verilog(code, self.timeout)
        if not res.available:
            return "lint unavailable"
        issues = res.errors + res.warnings
        return "no lint warnings" if not issues else "; ".join(
            f"line {w['line']} {w['code']}: {w['msg']}" for w in issues[:5])

    def _finish(self, code: str, action: str, n_tokens: int | None, info: dict):
        self.done = True
        n_tokens = approx_tokens(action) if n_tokens is None else n_tokens
        sim = self._simulate(code) if code else None
        reward, parts = verifiable_reward(sim, n_tokens, self.cfg)
        flags = detect_hacks(code, sim, self.task) if (self.detect and code) else []
        if flags and "hack_penalty" in self.cfg.safeguards:
            reward, parts = -self.cfg.hack_penalty, {**parts, "hack_penalty": True}
        info.update(
            code=code, submitted=True, n_tokens=n_tokens, reward_parts=parts,
            compiled=bool(sim and sim.compiled), pass_rate=sim.pass_rate if sim else 0.0,
            all_passed=bool(sim and sim.all_passed), tests=sim.tests if sim else {},
            timed_out=bool(sim and sim.timed_out), hack_flags=[f.to_dict() for f in flags],
            hacked=bool(flags),
        )
        return self._observation(), reward, True, info
