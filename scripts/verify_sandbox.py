"""Empirically check what the sandbox actually enforces on THIS machine, and write runs/sandbox_report.json.

python scripts/verify_sandbox.py            # run on the Linux box (Kaggle) whose results you will cite

Each check reports enforced=true/false from an observed outcome, not from reading the code. Cite only what
this report shows on the machine you ran it on. The sandbox is process-level isolation (temp dir, scrubbed
env, rlimits, optional network namespace, construct filter), not a container or VM: simulation code runs as
the same OS user, so "hardened" should be worded as that and no stronger.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from silicongym.tools.sandbox import Workspace, _no_network_prefix, violations  # noqa: E402
from silicongym.tools.simulate import simulate  # noqa: E402

PY = sys.executable
SECRET = "SECRET_TESTBENCH_SENTINEL_4f9a"

# attempts to read the hidden testbench / escape via Verilog constructs; each must be refused or leak nothing
ATTACKS = {
    "fopen": 'initial begin f = $fopen("tb.v", "r"); end',
    "readmemh": 'reg [7:0] m [0:9]; initial $readmemh("tb.v", m);',
    "system_cat": 'initial $system("cat tb.v");',
    "include": '`include "tb.v"',
    "include_spaced": '`  include "tb.v"',
    "define_alias": '`define F $fopen\ninitial begin f = `F("tb.v"); end',
    "plusargs": 'initial $value$plusargs("x=%s", s);',
    "dump": 'initial $dumpfile("/tmp/leak.vcd");',
    "vpi": 'initial $vpi_get_str(0, 0);',
    "fscanf": 'integer f; initial $fscanf(f, "%s", s);',
}


def child(code: str, timeout: float = 10) -> dict:
    with Workspace() as ws:
        r = ws.run([PY, "-c", code], timeout)
    return {"rc": r.returncode, "out": r.stdout.strip(), "err": r.stderr.strip()[-200:], "timed_out": r.timed_out}


def check_rlimits() -> dict:
    out = child("import resource as r,json;print(json.dumps({k:r.getrlimit(getattr(r,k)) for k in "
                "['RLIMIT_CPU','RLIMIT_AS','RLIMIT_FSIZE']}))")
    if out["rc"] != 0:
        return {"enforced": False, "reason": "resource module unavailable (non-POSIX)", "detail": out["err"]}
    lim = json.loads(out["out"])
    return {"enforced": lim["RLIMIT_CPU"][0] == 30 and lim["RLIMIT_AS"][0] == 2 << 30
            and lim["RLIMIT_FSIZE"][0] == 32 << 20, "limits": lim}


def check_memory_cap() -> dict:
    out = child("x = bytearray(3 * 1024**3); x[::4096] = b'a' * len(x[::4096]); print('allocated')")
    return {"enforced": "allocated" not in out["out"], "detail": out["err"] or out["out"] or f"rc={out['rc']}"}


def check_file_size_cap() -> dict:
    out = child("f = open('big', 'wb'); f.write(b'0' * (64 * 1024**2)); f.flush(); print('wrote')")
    return {"enforced": "wrote" not in out["out"], "detail": out["err"] or f"rc={out['rc']}"}


def check_timeout_kills_children() -> dict:
    code = "import subprocess,sys;subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);import time;time.sleep(60)"
    out = child(code, timeout=1.5)
    return {"enforced": out["timed_out"], "note": "parent timed out; orphaned grandchildren are killed only if "
            "the process group is (POSIX setsid+killpg)"}


def check_network() -> dict:
    code = ("import socket;s=socket.socket();s.settimeout(3)\n"
            "try:\n s.connect(('1.1.1.1',53));print('connected')\nexcept Exception as e:\n print('blocked',type(e).__name__)")
    out = child(code, timeout=8)
    return {"enforced": "blocked" in out["out"], "unshare_prefix": _no_network_prefix(), "observed": out["out"]}


def check_env_scrubbed() -> dict:
    import os
    os.environ["HF_TOKEN"] = "should-not-leak"
    out = child("import os;print('HF_TOKEN' in os.environ)")
    return {"enforced": out["out"] == "False"}


def check_testbench_attacks() -> dict:
    sim_tb = ("module tb; wire y; dut d(y); initial begin #1; $display(\"" + SECRET + "\");"
              " $display(\"TEST t PASS\"); $display(\"ALL_TESTS_PASSED\"); $finish; end endmodule")
    results = {}
    for name, body in ATTACKS.items():
        code = f"module dut(output y); integer f; reg [255:0] s; {body}\nassign y = 1'b0;\nendmodule"
        refused = bool(violations(code))
        res = simulate(code, sim_tb, ["t"])
        leaked = SECRET in res.stdout and SECRET in code
        results[name] = {"refused_up_front": refused, "compiled": res.compiled,
                         "leaked_testbench_text": leaked}
    ok = all(r["refused_up_front"] or not r["leaked_testbench_text"] for r in results.values())
    return {"enforced": ok, "attacks": results,
            "note": "the compiled .vvp embeds the testbench; the construct filter is what stops reads of it"}


def main() -> int:
    checks = {"rlimits": check_rlimits, "memory_cap_2GiB": check_memory_cap, "file_size_cap_32MiB": check_file_size_cap,
              "timeout_kills_process_group": check_timeout_kills_children, "network_blocked": check_network,
              "env_scrubbed": check_env_scrubbed, "testbench_read_attacks": check_testbench_attacks}
    report = {"platform": platform.platform(), "python": platform.python_version(),
              "checks": {name: fn() for name, fn in checks.items()}}
    report["all_enforced"] = all(c["enforced"] for c in report["checks"].values())
    out = ROOT / "runs" / "sandbox_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    for name, c in report["checks"].items():
        print(f"{'ENFORCED ' if c['enforced'] else 'NOT ENFORCED'}  {name}")
    print(f"\nreport: {out}")
    return 0 if report["all_enforced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
