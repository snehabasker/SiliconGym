import sys

import pytest

from silicongym.tools.compile import compile_verilog, parse_diagnostics
from silicongym.tools.lint import lint_verilog
from silicongym.tools.sandbox import Workspace, violations
from silicongym.tools.simulate import parse_tests, simulate

from conftest import needs_iverilog

ADDER = "module add(input [1:0] a, input [1:0] b, output [2:0] s); assign s = a + b; endmodule"
TB = """
module tb; reg [1:0] a, b; wire [2:0] s; add d(a, b, s);
  initial begin
    a = 1; b = 2; #1; if (s === 3) $display("TEST sum PASS"); else $display("TEST sum FAIL");
    a = 3; b = 3; #1; if (s === 6) $display("TEST carry PASS"); else $display("TEST carry FAIL");
    $display("ALL_TESTS_PASSED"); $finish;
  end
endmodule
"""


# ---- sandbox ----
@pytest.mark.parametrize("snippet", [
    'initial $fopen("tb.v");', '$readmemh("x", m);', '$system("ls");', '`include "tb.v"',
    '$dumpfile("a.vcd");', '$value$plusargs("a=%d", x);'])
def test_violations_detected(snippet):
    assert violations(f"module m; {snippet} endmodule")


def test_violations_ignore_comments_and_clean_code():
    assert violations("module m; // $fopen is not used here\n/* $system */ endmodule") == []
    assert violations(ADDER) == []


def test_workspace_timeout_returns_result_not_exception():
    with Workspace() as ws:
        res = ws.run([sys.executable, "-c", "while True: pass"], timeout=0.5)
    assert res.timed_out


def test_workspace_missing_binary_is_an_error_result():
    with Workspace() as ws:
        res = ws.run(["definitely-not-a-binary"])
    assert res.returncode == 127


def test_workspace_rejects_path_escape_and_cleans_up():
    with Workspace() as ws:
        with pytest.raises(ValueError):
            ws.write("../evil.txt", "x")
        d = ws.dir
    assert not d.exists()


def test_workspace_output_is_capped():
    with Workspace() as ws:
        res = ws.run([sys.executable, "-c", "print('x' * 500000)"])
    assert len(res.stdout) <= 64_000


# ---- compile ----
def test_parse_diagnostics():
    errs = parse_diagnostics("dut.v:3: error: Unknown module type: foo\ndut.v:4: warning: x\n")
    assert errs == [{"line": 3, "msg": "Unknown module type: foo"}]


def test_compile_refuses_forbidden_construct_without_running_a_tool():
    res = compile_verilog('module m; initial $system("rm -rf /"); endmodule')
    assert not res.ok and res.violations


@needs_iverilog
def test_compile_ok_and_error():
    assert compile_verilog(ADDER).ok
    bad = compile_verilog("module m(input a, output b); assign b = ; endmodule")
    assert not bad.ok and bad.errors


# ---- simulate ----
def test_parse_tests_missing_means_failed():
    out = "TEST a PASS\nTEST b FAIL\nTEST ghost PASS\n"
    assert parse_tests(out, ["a", "b", "c"]) == {"a": True, "b": False, "c": False}


@needs_iverilog
def test_simulate_pass_and_partial():
    ok = simulate(ADDER, TB, ["sum", "carry"])
    assert ok.all_passed and ok.pass_rate == 1.0
    half = simulate(ADDER.replace("a + b", "a | b"), TB, ["sum", "carry"])  # 1|2=3 ok, 3|3=3 != 6
    assert half.tests == {"sum": True, "carry": False} and not half.all_passed


@needs_iverilog
def test_simulate_early_finish_earns_nothing():
    cheat = "module add(input [1:0] a, input [1:0] b, output [2:0] s); initial $finish; endmodule"
    res = simulate(cheat, TB, ["sum", "carry"])
    assert res.pass_rate == 0 and not res.all_passed


@needs_iverilog
def test_simulate_timeout_is_an_observation():
    hang = "module add(input [1:0] a, input [1:0] b, output [2:0] s); initial forever begin end endmodule"
    res = simulate(hang, TB, ["sum", "carry"], timeout=2)
    assert res.timed_out and not res.all_passed


def test_simulate_rejects_file_access_before_any_simulator_runs():
    spy = 'module add(input a, output b); initial $fopen("tb.v"); endmodule'
    res = simulate(spy, TB, ["sum"])
    assert res.violations and not res.compiled


def test_lint_reports_unavailable_or_runs():
    assert lint_verilog(ADDER).available in (True, False)


@needs_iverilog
def test_non_ascii_and_lone_surrogate_model_output_never_crashes_a_tool():
    weird = "module m(output y); // 终结 éè \ud800\nassign y = 1'b0;\nendmodule"
    assert compile_verilog(weird).ok
    res = simulate(weird, TB, ["sum"])
    assert not res.all_passed          # scored as a normal failure, not an exception
