"""iverilog wrapper: compile a candidate module on its own and parse the diagnostics."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .sandbox import DEFAULT_TIMEOUT_S, Workspace, violations

_DIAG = re.compile(r"^(?P<file>[^:\n]+):(?P<line>\d+):\s*(?P<kind>error|warning|sorry):?\s*(?P<msg>.*)$", re.M)


@dataclass
class CompileResult:
    ok: bool
    stderr: str = ""
    errors: list[dict] = field(default_factory=list)  # {"line": int, "msg": str}
    violations: list[str] = field(default_factory=list)
    timed_out: bool = False


def parse_diagnostics(stderr: str) -> list[dict]:
    return [{"line": int(m["line"]), "msg": m["msg"].strip()}
            for m in _DIAG.finditer(stderr) if m["kind"] != "warning"]


def compile_verilog(code: str, timeout: float = DEFAULT_TIMEOUT_S) -> CompileResult:
    bad = violations(code)
    if bad:
        return CompileResult(False, f"forbidden constructs: {', '.join(bad)}", violations=bad)
    with Workspace() as ws:
        ws.write("dut.v", code)
        res = ws.run(["iverilog", "-g2012", "-o", "out.vvp", "dut.v"], timeout)
    if res.timed_out:
        return CompileResult(False, "compile timeout", timed_out=True)
    return CompileResult(res.returncode == 0, res.stderr, parse_diagnostics(res.stderr))
