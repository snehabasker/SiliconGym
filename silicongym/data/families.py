"""Parametrised Verilog task families. Each family function yields Task objects (split unset).

SPLITS assigns whole families to a split: width/polarity variants of one design are near-duplicates
by construction, so splitting inside a family would leak. The five families that SiliconLM's own
DPO pairs were sampled from (adder, mux2, dff, counter, priority encoder) are train-only, so the
SFT/DPO baselines are never evaluated on prompts they were trained on.
"""

from __future__ import annotations

from math import ceil, log2
from typing import Iterator

from .task import Task
from .testbench import STEP_NEG, Port, build_tb, header, ones, rnd, sweep

WIDTHS = [2, 3, 4, 5, 6, 8]
ST = "STEP;"


def _task(family: str, tag: str, params: dict, spec: str, inputs, outputs,
          golden: str, groups, **tb_kwargs) -> Task:
    mod = f"{family}_{tag}"
    hdr = header(mod, inputs, outputs)
    return Task(
        id=mod, family=family, params=params,
        spec=f"{spec}\nModule header:\n{hdr}\nReply with the complete Verilog module.",
        module_name=mod, module_header=hdr,
        testbench=build_tb(mod, inputs, outputs, golden, groups, **tb_kwargs),
        test_names=[g[0] for g in groups], reference=golden.replace("__MOD__", mod),
    )


def _golden(inputs, outputs, body: str) -> str:
    return header("__MOD__", inputs, outputs) + "\n" + body + "\nendmodule\n"


# ---- train families ---------------------------------------------------------------------------

def adder() -> Iterator[Task]:
    for n in WIDTHS:
        i, o = [Port("a", n), Port("b", n)], [Port("sum", n + 1)]
        g = _golden(i, o, "  assign sum = a + b;")
        groups = [
            ("zero_operand", f"a = 0; b = 0; #1; check; b = {ones(n)}; #1; check; a = {ones(n)}; b = 0; #1; check;"),
            ("carry_out", f"a = {ones(n)}; b = 1; #1; check; b = {ones(n)}; #1; check;"),
            ("random_1", rnd(["a", "b"])), ("random_2", rnd(["a", "b"])),
        ]
        yield _task("adder", f"w{n}", {"width": n},
                    f"Implement a {n}-bit unsigned adder with carry out. sum must equal a + b "
                    f"({n + 1}-bit result). Combinational logic only.", i, o, g, groups)


def mux2() -> Iterator[Task]:
    for n in WIDTHS:
        i, o = [Port("d0", n), Port("d1", n), Port("sel")], [Port("y", n)]
        g = _golden(i, o, "  assign y = sel ? d1 : d0;")
        groups = [("sel_0", "sel = 0; " + rnd(["d0", "d1"], 16)),
                  ("sel_1", "sel = 1; " + rnd(["d0", "d1"], 16)),
                  ("toggle", rnd(["d0", "d1", "sel"], 32))]
        yield _task("mux2", f"w{n}", {"width": n},
                    f"Implement a 2-to-1 multiplexer for {n}-bit data: y = d1 when sel is 1, else d0.",
                    i, o, g, groups)


def mux4() -> Iterator[Task]:
    for n in WIDTHS:
        i = [Port(f"d{k}", n) for k in range(4)] + [Port("sel", 2)]
        o = [Port("y", n)]
        g = _golden(i, o, "  assign y = sel == 0 ? d0 : sel == 1 ? d1 : sel == 2 ? d2 : d3;")
        data = ["d0", "d1", "d2", "d3"]
        groups = [(f"sel_{k}", f"sel = {k}; " + rnd(data, 16)) for k in range(4)]
        yield _task("mux4", f"w{n}", {"width": n},
                    f"Implement a 4-to-1 multiplexer for {n}-bit data: y = d0, d1, d2 or d3 when sel is "
                    "0, 1, 2 or 3 respectively.", i, o, g, groups)


def decoder() -> Iterator[Task]:
    for k in [1, 2, 3, 4]:
        w = 2 ** k
        i, o = [Port("in", k), Port("en")], [Port("out", w)]
        g = _golden(i, o, f"  assign out = en ? ({w}'d1 << in) : {w}'d0;")
        groups = [("disabled", "en = 0; " + sweep("in", 0, w)),
                  ("enabled_low", "en = 1; " + sweep("in", 0, w // 2)),
                  ("enabled_high", "en = 1; " + sweep("in", w // 2, w))]
        yield _task("decoder", f"k{k}", {"k": k},
                    f"Implement a {k}-to-{w} decoder with enable. When en is 1, bit `in` of out is 1 and "
                    "every other bit is 0. When en is 0, out is all zeros.", i, o, g, groups)


def comparator() -> Iterator[Task]:
    for n in WIDTHS:
        i, o = [Port("a", n), Port("b", n)], [Port("eq"), Port("lt"), Port("gt")]
        g = _golden(i, o, "  assign eq = a == b;\n  assign lt = a < b;\n  assign gt = a > b;")
        fix = "if (a == b) b = b + 1;"
        groups = [
            ("equal", rnd(["a"], 16, "b = a;")),
            ("a_less", rnd(["a", "b"], 24, f"{fix} if (a > b) begin tmp = a; a = b; b = tmp; end")),
            ("a_greater", rnd(["a", "b"], 24, f"{fix} if (a < b) begin tmp = a; a = b; b = tmp; end")),
            ("extremes", f"a = 0; b = {ones(n)}; #1; check; a = {ones(n)}; b = 0; #1; check; "
                         f"a = {ones(n)}; b = {ones(n)}; #1; check; a = 0; b = 0; #1; check;"),
        ]
        yield _task("comparator", f"w{n}", {"width": n},
                    f"Implement an {n}-bit unsigned magnitude comparator. eq is 1 when a == b, lt is 1 "
                    "when a < b, gt is 1 when a > b. Exactly one output is 1.",
                    i, o, g, groups, extra_decl="reg [31:0] tmp;")


def parity() -> Iterator[Task]:
    for n in WIDTHS:
        i, o = [Port("in", n)], [Port("p")]
        g = _golden(i, o, "  assign p = ^in;")
        groups = [("zero_and_ones", f"in = 0; #1; check; in = {ones(n)}; #1; check;"),
                  ("single_bit", f"for (i = 0; i < {n}; i = i + 1) begin in = 1 << i; #1; check; end"),
                  ("random", rnd(["in"], 40))]
        yield _task("parity", f"w{n}", {"width": n},
                    f"Implement a parity generator for a {n}-bit input: p is 1 when an odd number of "
                    "bits in `in` are 1, otherwise 0.", i, o, g, groups)


def prienc() -> Iterator[Task]:
    for n in [2, 4, 8, 16]:
        k = int(log2(n))
        i, o = [Port("in", n)], [Port("out", k, reg=True), Port("valid", reg=True)]
        g = _golden(i, o, f"  integer j;\n  always @(*) begin\n    valid = |in; out = 0;\n"
                          f"    for (j = 0; j < {n}; j = j + 1) if (in[j]) out = j;\n  end")
        groups = [("none_set", "in = 0; #1; check;"),
                  ("single_hot", f"for (i = 0; i < {n}; i = i + 1) begin in = 1 << i; #1; check; end"),
                  ("multi_hot", rnd(["in"], 40))]
        yield _task("prienc", f"n{n}", {"n": n},
                    f"Implement a {n}-to-{k} priority encoder with valid flag. The highest set bit of "
                    f"`in` wins (bit {n - 1} is highest) and out is its index. valid is 1 when any "
                    "input bit is set; otherwise valid is 0 and out is 0.", i, o, g, groups)


def dff() -> Iterator[Task]:
    for is_async in (True, False):
        for active_high in (True, False):
            for has_en in (False, True):
                rst = "rst" if active_high else "rst_n"
                act, idle = ("1" if active_high else "0"), ("0" if active_high else "1")
                i = [Port("clk"), Port(rst)] + ([Port("en")] if has_en else []) + [Port("d", 4)]
                o = [Port("q", 4, reg=True)]
                edge = (f"posedge clk or {'posedge' if active_high else 'negedge'} {rst}"
                        if is_async else "posedge clk")
                load = "else if (en) q <= d;" if has_en else "else q <= d;"
                g = _golden(i, o, f"  always @({edge})\n    if ({rst} == 1'b{act}) q <= 4'd0;\n    {load}")
                en_on = "en = 1; " if has_en else ""
                groups = [
                    ("reset", f"{rst} = {act}; {en_on}d = 4'd9; {ST} {ST}"),
                    ("capture", f"{rst} = {idle}; {en_on}" + rnd(["d"], 12, step=ST)),
                    ("reset_midcycle", f"@(negedge clk); {rst} = {act}; #1; check; {rst} = {idle}; "
                                       f"@(posedge clk); #1; check; {ST}"),
                ]
                if has_en:
                    groups.insert(2, ("hold", "en = 0; " + rnd(["d"], 8, step=ST) + " en = 1;"))
                tag = f"{'async' if is_async else 'sync'}_{'hi' if active_high else 'lo'}{'_en' if has_en else ''}"
                kind = "asynchronous" if is_async else "synchronous"
                pol = "high" if active_high else "low"
                yield _task("dff", tag, {"async": is_async, "active_high": active_high, "enable": has_en},
                            f"Implement a 4-bit D register with active-{pol} {kind} reset `{rst}`"
                            f"{' and a load enable `en`' if has_en else ''}. On reset q becomes 0"
                            f"{' immediately' if is_async else ' at the next rising clock edge'}; otherwise q "
                            f"captures d on the rising edge of clk{' only when en is 1' if has_en else ''}.",
                            i, o, g, groups, clocked=True)


def _counter(family: str, up_down: bool) -> Iterator[Task]:
    for n in WIDTHS:
        i = [Port("clk"), Port("rst"), Port("en")] + ([Port("up")] if up_down else [])
        o = [Port("count", n, reg=True)]
        step = "(up ? count + 1'b1 : count - 1'b1)" if up_down else "count + 1'b1"
        g = _golden(i, o, f"  always @(posedge clk)\n    if (rst) count <= 0;\n    else if (en) count <= {step};")
        dirn = "up = 1; " if up_down else ""
        groups = [
            ("reset", f"rst = 1; en = 0; {dirn}{ST} {ST}"),
            ("count", f"rst = 0; en = 1; {dirn}repeat (5) begin {ST} end"),
            ("hold", f"en = 0; repeat (3) begin {ST} end en = 1;"),
            ("wrap", f"repeat ({2 ** n + 2}) begin {ST} end"),
            ("reset_mid", f"rst = 1; {ST} rst = 0; repeat (3) begin {ST} end"),
            ("random", f"for (i = 0; i < 60; i = i + 1) begin rst = (($random(seed) & 15) == 0); "
                       f"en = $random(seed); {'up = $random(seed); ' if up_down else ''}{ST} end"),
        ]
        if up_down:
            groups.insert(3, ("count_down", f"rst = 1; {ST} rst = 0; en = 1; up = 0; "
                                            f"repeat ({2 ** n + 2}) begin {ST} end"))
        direction = "counts up when up is 1 and down when up is 0" if up_down else "increments"
        spec = (f"Implement a {n}-bit {'up/down' if up_down else 'up'}-counter with synchronous "
                f"active-high reset `rst` and enable `en`. On the rising clock edge: count resets to 0 if rst, "
                f"otherwise {direction} when en is 1 and holds when en is 0. It wraps around at the "
                f"{n}-bit limits.")
        yield _task(family, f"w{n}", {"width": n}, spec, i, o, g, groups, clocked=True)


def counter() -> Iterator[Task]:
    yield from _counter("counter", False)


def updown() -> Iterator[Task]:
    yield from _counter("updown", True)


def shiftreg() -> Iterator[Task]:
    for n in WIDTHS:
        i, o = [Port("clk"), Port("rst"), Port("si")], [Port("q", n, reg=True)]
        g = _golden(i, o, f"  always @(posedge clk)\n    if (rst) q <= 0;\n    else q <= {{q[{n - 2}:0], si}};")
        groups = [
            ("reset", f"rst = 1; si = 1; {ST} {ST}"),
            ("fill_ones", f"rst = 0; si = 1; repeat ({n + 1}) begin {ST} end"),
            ("fill_zeros", f"si = 0; repeat ({n + 1}) begin {ST} end"),
            ("pattern", f"for (i = 0; i < {2 * n}; i = i + 1) begin si = i[0]; {ST} end"),
            ("reset_mid", f"si = 1; {ST} {ST} rst = 1; {ST} rst = 0; repeat (3) begin {ST} end"),
            ("random", f"for (i = 0; i < 40; i = i + 1) begin si = $random(seed); {ST} end"),
        ]
        yield _task("shiftreg", f"w{n}", {"width": n},
                    f"Implement a {n}-bit serial-in parallel-out shift register with synchronous "
                    "active-high reset `rst`. On each rising clock edge q resets to 0 if rst, otherwise "
                    "shifts left by one with si entering at bit 0.", i, o, g, groups, clocked=True)


# ---- validation families ----------------------------------------------------------------------

def gray() -> Iterator[Task]:
    for decode in (False, True):
        for n in [3, 4, 5, 8]:
            src, dst = ("g", "b") if decode else ("b", "g")
            i, o = [Port(src, n)], [Port(dst, n, reg=decode)]
            if decode:
                body = (f"  integer j;\n  always @(*) begin\n    {dst}[{n - 1}] = {src}[{n - 1}];\n"
                        f"    for (j = {n - 2}; j >= 0; j = j - 1) {dst}[j] = {dst}[j + 1] ^ {src}[j];\n  end")
            else:
                body = "  assign g = b ^ (b >> 1);"
            g = _golden(i, o, body)
            half = 2 ** n // 2
            groups = [("low_half", sweep(src, 0, half)), ("high_half", sweep(src, half, 2 ** n)),
                      ("walking_ones", f"for (i = 0; i < {n}; i = i + 1) begin {src} = 1 << i; #1; check; end")]
            what = "Gray code to binary" if decode else "binary to Gray code"
            yield _task("gray", f"{'dec' if decode else 'enc'}_w{n}", {"width": n, "decode": decode},
                        f"Implement a {n}-bit {what} converter (combinational), using the standard "
                        "reflected Gray code where consecutive values differ in one bit.", i, o, g, groups)


def edge() -> Iterator[Task]:
    cond = {"rising": "sig & ~prev", "falling": "~sig & prev", "both": "sig ^ prev"}
    desc = {"rising": "a 0-to-1 transition", "falling": "a 1-to-0 transition", "both": "any change"}
    for kind in cond:
        for registered in (False, True):
            i, o = [Port("clk"), Port("rst"), Port("sig")], [Port("pulse", reg=registered)]
            prev = "  reg prev;\n  always @(posedge clk) prev <= rst ? 1'b0 : sig;\n"
            if registered:
                body = prev + f"  always @(posedge clk) pulse <= rst ? 1'b0 : ({cond[kind]});"
            else:
                body = prev + f"  assign pulse = {cond[kind]};"
            g = _golden(i, o, body)

            def seq(bits: str) -> str:
                return " ".join(f"sig = {b}; {ST}" for b in bits)
            groups = [("reset", f"rst = 1; sig = 0; {ST} {ST} rst = 0;"),
                      ("rise_fall", seq("00110011")), ("toggle", seq("10101010")),
                      ("random", f"for (i = 0; i < 40; i = i + 1) begin sig = $random(seed); {ST} end")]
            tag = f"{kind}_{'reg' if registered else 'comb'}"
            expr = cond[kind].replace("prev", "prev").replace("sig", "sig")
            how = (f"pulse is a registered output: on each rising clock edge it loads ({expr}), "
                   "or 0 if rst" if registered else f"pulse = {expr} (combinational)")
            yield _task("edge", tag, {"kind": kind, "registered": registered},
                        f"Implement an edge detector for {desc[kind]} on `sig`, with synchronous "
                        "active-high reset `rst`. Keep `prev`, the value of sig at the previous rising clock "
                        f"edge (0 after reset). {how}.",
                        i, o, g, groups, clocked=True, step=STEP_NEG)


# ---- test families (not seen by SiliconLM's DPO pairs) ----------------------------------------

def barrel() -> Iterator[Task]:
    for rotate in (False, True):
        for n in [4, 8, 16]:
            s = int(log2(n))
            i, o = [Port("in", n), Port("shamt", s), Port("dir")], [Port("out", n)]
            if rotate:
                body = (f"  assign out = dir ? ((in >> shamt) | (in << ({n} - shamt)))"
                        f" : ((in << shamt) | (in >> ({n} - shamt)));")
            else:
                body = "  assign out = dir ? (in >> shamt) : (in << shamt);"
            g = _golden(i, o, body)
            groups = [("no_shift", "shamt = 0; dir = 0; " + rnd(["in"], 8)),
                      ("left", "dir = 0; " + rnd(["in", "shamt"], 24)),
                      ("right", "dir = 1; " + rnd(["in", "shamt"], 24)),
                      ("random", rnd(["in", "shamt", "dir"], 40))]
            op = "rotator" if rotate else "logical shifter"
            beh = "bits shifted out re-enter at the other end" if rotate else "vacated bits are filled with 0"
            yield _task("barrel", f"{'rot' if rotate else 'shift'}_w{n}", {"width": n, "rotate": rotate},
                        f"Implement a combinational {n}-bit barrel {op}. dir = 0 shifts `in` left by "
                        f"`shamt` positions and dir = 1 shifts it right; {beh}.", i, o, g, groups)


def popcount() -> Iterator[Task]:
    for n in WIDTHS:
        k = ceil(log2(n + 1))
        i, o = [Port("in", n)], [Port("count", k, reg=True)]
        g = _golden(i, o, f"  integer j;\n  always @(*) begin\n    count = 0;\n"
                          f"    for (j = 0; j < {n}; j = j + 1) count = count + in[j];\n  end")
        groups = [("zero_and_ones", f"in = 0; #1; check; in = {ones(n)}; #1; check;"),
                  ("single_bit", f"for (i = 0; i < {n}; i = i + 1) begin in = 1 << i; #1; check; end"),
                  ("random", rnd(["in"], 40))]
        yield _task("popcount", f"w{n}", {"width": n},
                    f"Implement a population counter: count is the number of 1 bits in the {n}-bit "
                    f"input `in` ({k}-bit result).", i, o, g, groups)


def satadd() -> Iterator[Task]:
    for n in WIDTHS:
        i, o = [Port("a", n), Port("b", n)], [Port("y", n)]
        g = _golden(i, o, f"  wire [{n}:0] t = a + b;\n  assign y = t[{n}] ? {ones(n)} : t[{n - 1}:0];")
        top, mask = 1 << (n - 1), (1 << (n - 1)) - 1
        groups = [
            ("no_overflow", rnd(["a", "b"], 24, f"a = a & {n}'d{mask}; b = b & {n}'d{mask};")),
            ("overflow", rnd(["a", "b"], 24, f"a = a | {n}'d{top}; b = b | {n}'d{top};")),
            ("boundary", f"a = {ones(n)}; b = 0; #1; check; b = 1; #1; check; a = 0; b = {ones(n)}; #1; check; "
                         f"a = {ones(n)}; b = {ones(n)}; #1; check;"),
            ("random", rnd(["a", "b"], 40)),
        ]
        yield _task("satadd", f"w{n}", {"width": n},
                    f"Implement an {n}-bit unsigned saturating adder: y = a + b, but if the true sum "
                    f"exceeds {n} bits, y is clamped to the all-ones value instead of wrapping.",
                    i, o, g, groups)


FAMILIES = {f.__name__: f for f in [adder, mux2, mux4, decoder, comparator, parity, prienc, dff,
                                    counter, shiftreg, gray, edge, barrel, popcount, updown, satadd]}
SPLITS = {
    "train": ["adder", "mux2", "mux4", "decoder", "comparator", "parity", "prienc", "dff", "counter", "shiftreg"],
    "val": ["gray", "edge"],
    "test": ["barrel", "popcount", "updown", "satadd"],
}
