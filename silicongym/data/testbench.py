"""Generic testbench builder: drive the DUT and a hidden golden model with the same stimulus.

A family supplies ports, a golden module (with the placeholder __MOD__ as its name) and a list of
(group_name, stimulus) pairs. Stimulus is Verilog using `check;` (compare DUT vs golden outputs)
and, for clocked designs, `STEP;`-style helpers from the `step` argument. Each group becomes one
`TEST <group> PASS|FAIL` line, which is what per-test partial credit is computed from.
"""

from __future__ import annotations

from dataclasses import dataclass

# Step styles for clocked designs. Inputs change 1ns after a posedge (clk period 10). Each step also
# samples *before* the next posedge, where a design clocked on the wrong edge has already changed
# state but the golden model has not. STEP_POS suits registered outputs; STEP_NEG also samples early
# in the cycle (t+4) for outputs that depend combinationally on the current inputs.
STEP_POS = "#8; check; @(posedge clk); #1; check;"
STEP_NEG = "#3; check; #5; check; @(posedge clk); #1;"


@dataclass
class Port:
    name: str
    width: int = 1
    reg: bool = False  # outputs only: declared `output reg`


def rng(w: int) -> str:
    return f"[{w - 1}:0] " if w > 1 else ""


def ones(n: int) -> str:
    return "{%d{1'b1}}" % n


def header(mod: str, inputs: list[Port], outputs: list[Port]) -> str:
    ports = [f"input {rng(p.width)}{p.name}" for p in inputs]
    ports += [f"output {'reg ' if p.reg else ''}{rng(p.width)}{p.name}" for p in outputs]
    return f"module {mod}({', '.join(ports)});"


def rnd(names: list[str], n: int = 32, extra: str = "", step: str = "#1; check;") -> str:
    drive = " ".join(f"{v} = $random(seed);" for v in names)
    return f"for (i = 0; i < {n}; i = i + 1) begin {drive} {extra} {step} end"


def sweep(var: str, lo: int, hi: int, extra: str = "") -> str:
    return f"for (i = {lo}; i < {hi}; i = i + 1) begin {var} = i; {extra} #1; check; end"


def build_tb(mod: str, inputs: list[Port], outputs: list[Port], golden: str,
             groups: list[tuple[str, str]], clocked: bool = False, step: str = STEP_POS,
             extra_decl: str = "") -> str:
    data_in = [p for p in inputs if not (clocked and p.name == "clk")]
    conn = ", ".join(f".{p.name}({p.name})" for p in inputs)
    lines = [golden.replace("__MOD__", "golden_ref"), "module tb;"]
    lines += [f"  reg {rng(p.width)}{p.name};" for p in data_in]
    if clocked:
        lines += ["  reg clk;", "  always #5 clk = ~clk;"]
    lines += [f"  wire {rng(p.width)}{p.name};" for p in outputs]
    lines += [f"  wire {rng(p.width)}gold_{p.name};" for p in outputs]
    lines += ["  integer i, err, bad, seed;", extra_decl]
    lines.append(f"  {mod} dut({conn}, " + ", ".join(f".{p.name}({p.name})" for p in outputs) + ");")
    lines.append(f"  golden_ref gold({conn}, " +
                 ", ".join(f".{p.name}(gold_{p.name})" for p in outputs) + ");")
    dut_cat = ", ".join(p.name for p in outputs)
    gold_cat = ", ".join(f"gold_{p.name}" for p in outputs)
    lines.append(f"  task check; begin if ({{{dut_cat}}} !== {{{gold_cat}}}) err = err + 1; end endtask")
    lines.append("  initial begin")
    lines.append("    seed = 12345; err = 0; bad = 0;")
    lines += [f"    {p.name} = 0;" for p in data_in]
    if clocked:
        lines += ["    clk = 0;", "    @(posedge clk); #1;"]
    for name, stim in groups:
        stim = stim.replace("STEP;", step)
        lines.append(f"    err = 0;\n    {stim}")
        lines.append(f'    if (err == 0) $display("TEST {name} PASS@NONCE@"); '
                     f'else begin $display("TEST {name} FAIL"); bad = bad + 1; end')
    lines += ['    if (bad == 0) $display("ALL_TESTS_PASSED@NONCE@");', "    $finish;", "  end", "endmodule", ""]
    return "\n".join(lines)
