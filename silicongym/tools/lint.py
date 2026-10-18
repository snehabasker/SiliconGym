"""verilator --lint-only wrapper. Optional: returns available=False when verilator is missing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .sandbox import DEFAULT_TIMEOUT_S, Workspace, find_tool, violations

_WARN = re.compile(r"^%(?P<kind>Warning|Error)(?:-(?P<code>\w+))?: dut\.v:(?P<line>\d+):(?:\d+:)?\s*(?P<msg>.*)$", re.M)


@dataclass
class LintResult:
    available: bool
    warnings: list[dict] = field(default_factory=list)  # {"line", "code", "msg"}
    errors: list[dict] = field(default_factory=list)


def lint_verilog(code: str, timeout: float = DEFAULT_TIMEOUT_S) -> LintResult:
    if find_tool("verilator") is None:
        return LintResult(available=False)
    if violations(code):
        return LintResult(available=True, errors=[{"line": 0, "code": "FORBIDDEN", "msg": "forbidden constructs"}])
    with Workspace() as ws:
        ws.write("dut.v", code)
        res = ws.run(["verilator", "--lint-only", "-Wall", "dut.v"], timeout)
    out = LintResult(available=True)
    for m in _WARN.finditer(res.stderr):
        item = {"line": int(m["line"]), "code": m["code"] or "", "msg": m["msg"].strip()}
        (out.errors if m["kind"] == "Error" else out.warnings).append(item)
    return out
