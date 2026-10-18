"""Minimal second domain: write a Python function, scored by hidden unit tests.

Exists to show that `Env` is domain-agnostic and that `verifiable_reward` is reused unchanged.
Isolation is weaker than for Verilog: the solution runs in the sandbox's temp dir with a timeout
and scrubbed env, but Python code can read its own source tree, so this env is not hardened
against testbench peeking and must not be used for the hacking study.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass

from silicongym.rewards.verifiable import RewardConfig, verifiable_reward
from silicongym.tools.sandbox import DEFAULT_TIMEOUT_S, Workspace
from silicongym.tools.simulate import SimResult, parse_tests

from .base import Env

_FINAL = re.compile(r"<final>(.*?)</final>", re.S)
_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)

_RUNNER = """
import json, sys
sys.path.insert(0, ".")
import solution
cases = json.load(open("cases.json"))
for name, expr, expected in cases:
    try:
        ok = eval(expr, vars(solution)) == expected
    except Exception:
        ok = False
    print("TEST", name, "PASS" if ok else "FAIL")
"""


@dataclass
class PythonTask:
    id: str
    spec: str
    cases: list[tuple[str, str, object]]  # (test name, expression calling the function, expected)


class PythonEnv(Env):
    def __init__(self, reward_cfg: RewardConfig | None = None, timeout: float = DEFAULT_TIMEOUT_S):
        self.cfg, self.timeout = reward_cfg or RewardConfig(), timeout

    def reset(self, task: PythonTask) -> str:
        self.task, self.done = task, False
        return f"Task:\n{task.spec}\nReply with <final>python code</final>."

    def step(self, action: str, n_tokens: int | None = None) -> tuple[str, float, bool, dict]:
        assert not self.done, "episode finished; call reset()"
        self.done = True
        m = _FINAL.search(action)
        code = (m.group(1) if m else action).strip()
        if f := _FENCE.search(code):
            code = f.group(1).strip()
        sim = self._run(code)
        reward, parts = verifiable_reward(sim, n_tokens if n_tokens is not None else len(action.split()), self.cfg)
        return "", reward, True, {"turn": 1, "code": code, "reward_parts": parts,
                                  "tests": sim.tests, "pass_rate": sim.pass_rate}

    def _run(self, code: str) -> SimResult:
        names = [c[0] for c in self.task.cases]
        none = {n: False for n in names}
        try:
            compile(code, "solution.py", "exec")
        except SyntaxError as e:
            return SimResult(False, none, error=str(e))
        with Workspace() as ws:
            ws.write("solution.py", code)
            ws.write("runner.py", _RUNNER)
            ws.write("cases.json", json.dumps(self.task.cases))
            res = ws.run([sys.executable, "runner.py"], self.timeout)
        tests = parse_tests(res.stdout, names)
        return SimResult(True, tests, all_passed=all(tests.values()) and not res.timed_out,
                         stdout=res.stdout, error="timeout" if res.timed_out else res.stderr[-300:],
                         timed_out=res.timed_out)
