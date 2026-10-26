"""Synthesise (Yosys) and place (OpenROAD) a set of Verilog designs on sky130_fd_sc_hd and record the metrics.

python eda_run.py --manifest designs.jsonl --out results/golden [--period 5.0] [--no-pnr] [--jobs 2]

Manifest: one JSON object per line with
  design_id, task_id, kind ("golden" | "generated"), top, is_seq, clk_port, code,
  ref_code (optional: golden code, enables the formal equivalence check against it)

Per design the output directory holds the exact inputs, the tool logs and a result.json whose numbers are
parsed from those logs, so every figure in the summaries can be traced back to a raw file.
Standard library only; tool paths come from env vars (see tool_paths()).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENVROOT = Path(os.environ.get("EDA_ENVROOT", Path.home() / "eda" / "mamba" / "envs"))
PDK = Path(os.environ.get("EDA_PDK", ENVROOT / "pdk" / "share" / "pdk" / "sky130A"))


def dont_use_args() -> str:
    """-dont_use flags: the PDK's drc_exclude.cells plus low-power-flow, probe, clock, delay and physical-only cells.

    The PDK's no_synth.cells was tried and rejected: it bans nearly every _1 drive-strength cell
    (nand2_1, xor2_1, ...), which would inflate every design's area."""
    cells = ["sky130_fd_sc_hd__lpflow_*", "sky130_fd_sc_hd__probe*", "sky130_fd_sc_hd__clkbuf_*",
             "sky130_fd_sc_hd__clkinv*", "sky130_fd_sc_hd__clkdlybuf*", "sky130_fd_sc_hd__dlygate*",
             "sky130_fd_sc_hd__dlymetal*", "sky130_fd_sc_hd__decap_*", "sky130_fd_sc_hd__fill_*",
             "sky130_fd_sc_hd__tap*", "sky130_fd_sc_hd__diode_*"]
    p = PDK / "libs.tech" / "openlane" / "sky130_fd_sc_hd" / "drc_exclude.cells"
    cells += [l.strip() for l in p.read_text().splitlines() if l.strip()]
    return " ".join(f"-dont_use {c}" for c in dict.fromkeys(cells))


def tool_paths() -> dict:
    hd = PDK / "libs.ref" / "sky130_fd_sc_hd"
    return {
        "yosys": os.environ.get("YOSYS", str(ENVROOT / "eda" / "bin" / "yosys")),
        "openroad": os.environ.get("OPENROAD", str(ENVROOT / "orlh" / "bin" / "openroad")),
        "lib": str(hd / "lib" / "sky130_fd_sc_hd__tt_025C_1v80.lib"),
        "tlef": str(hd / "techlef" / "sky130_fd_sc_hd__nom.tlef"),
        "lef": str(hd / "lef" / "sky130_fd_sc_hd.lef"),
    }


def run(cmd: list[str], cwd: Path, log: Path, env: dict | None = None, timeout: int = 600) -> int:
    with log.open("w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        try:
            p = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                               env={**os.environ, **(env or {})})
            return p.returncode
        except subprocess.TimeoutExpired:
            f.write("\nTIMEOUT\n")
            return -9


YOSYS_SYNTH = """\
read_verilog -sv design.v
hierarchy -check -top {top}
synth -top {top} -flatten
dfflibmap -liberty {lib} {dont_use}
abc -liberty {lib} {dont_use}
opt_clean -purge
hilomap -singleton -hicell sky130_fd_sc_hd__conb_1 HI -locell sky130_fd_sc_hd__conb_1 LO
insbuf -buf sky130_fd_sc_hd__buf_2 A X
opt_clean -purge
tee -o stat.txt stat -liberty {lib}
tee -o stat.json stat -liberty {lib} -json
write_verilog -noattr -noexpr -nohex -nodec netlist.v
"""

# Bounded/unbounded formal equivalence of the generated module against the golden one.
# Combinational: SAT proof over all inputs. Sequential: 20-cycle bounded proof from the all-zero state.
YOSYS_EQUIV = """\
read_verilog -sv gold.v
prep -top {top}
design -stash gold
read_verilog -sv design.v
prep -top {top}
design -stash gate
design -copy-from gold -as gold {top}
design -copy-from gate -as gate {top}
miter -equiv -flatten -make_assert gold gate miter
hierarchy -top miter
{sat}
"""


def parse_stat(path: Path) -> dict:
    d = json.loads(path.read_text())
    top = d.get("design") or {}
    if "area" not in top and d.get("modules"):
        top = next(iter(d["modules"].values()))
    cells = {k: v for k, v in (top.get("num_cells_by_type") or {}).items()
             if k.startswith("sky130_")}
    return {"cells": int(top["num_cells"]), "area_um2": float(top["area"]), "cell_types": cells}


def num(rx: str, text: str):
    m = re.search(rx, text, re.M)
    return float(m.group(1)) if m else None


def parse_pnr(report: Path, period: float) -> dict:
    t = report.read_text()
    sec = lambda a, b: t.split(f"--- {a}\n", 1)[1].split(f"--- {b}\n", 1)[0] if f"--- {a}\n" in t else ""
    area_s, slack_s, crit_s, pow_s = sec("design_area", "cell_type_report"), sec("worst_slack_max", "worst_slack_min"), \
        sec("critical_path", "power"), t.split("--- power\n", 1)[-1]
    arrivals = re.findall(r"^\s*([-\d.]+)\s+data arrival time", crit_s, re.M)
    total = re.search(r"^Total\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", pow_s, re.M)
    return {
        "place_area_um2": num(r"Design area ([\d.]+)", area_s),
        "worst_slack_ns": num(r"worst slack (?:max )?([-\d.]+)", slack_s),
        "data_arrival_ns": float(arrivals[0]) if arrivals else None,   # first match = the path; the 2nd is its negation
        "period_ns": period,
        "power_internal_w": float(total.group(1)) if total else None,
        "power_switching_w": float(total.group(2)) if total else None,
        "power_leakage_w": float(total.group(3)) if total else None,
        "power_total_w": float(total.group(4)) if total else None,
    }


def process(item: dict, out: Path, tools: dict, period: float, do_pnr: bool) -> dict:
    d = out / "designs" / item["design_id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "design.v").write_text(item["code"])
    (d / "input.json").write_text(json.dumps({k: v for k, v in item.items() if k not in ("code", "ref_code")}, indent=1))
    res: dict = {"design_id": item["design_id"], "task_id": item["task_id"], "kind": item["kind"],
                 "top": item["top"], "is_seq": item["is_seq"], "status": "ok"}
    t0 = time.time()
    (d / "synth.ys").write_text(YOSYS_SYNTH.format(top=item["top"], lib=tools["lib"], dont_use=dont_use_args()))
    rc = run([tools["yosys"], "-q", "-l", "yosys_full.log", "synth.ys"], d, d / "yosys.log")
    if rc != 0 or not (d / "stat.json").exists():
        return {**res, "status": "synth_failed", "detail": (d / "yosys.log").read_text()[-600:]}
    res.update(parse_stat(d / "stat.json"))

    if item.get("ref_code"):
        (d / "gold.v").write_text(item["ref_code"])
        sat = ("sat -verify -prove-asserts -seq 20 -set-init-zero -show-inputs -show-outputs miter"
               if item["is_seq"] else "sat -verify -prove-asserts -show-inputs -show-outputs miter")
        (d / "equiv.ys").write_text(YOSYS_EQUIV.format(top=item["top"], sat=sat))
        rc = run([tools["yosys"], "-l", "equiv_full.log", "equiv.ys"], d, d / "equiv.log")
        full = (d / "equiv_full.log").read_text() if (d / "equiv_full.log").exists() else ""
        res["equiv_kind"] = "bounded_20_cycles" if item["is_seq"] else "combinational_sat_proof"
        # `sat -verify` exits non-zero when the proof fails and prints "SUCCESS!" when no counterexample exists
        res["equiv_to_golden"] = rc == 0 and "SUCCESS!" in full
        res["equiv_rc"] = rc

    if do_pnr:
        env = {"TOP": item["top"], "NETLIST": str(d / "netlist.v"), "OUTDIR": str(d),
               "PERIOD_NS": str(period), "IS_SEQ": "1" if item["is_seq"] else "0",
               "CLK_PORT": item.get("clk_port") or "clk", "LIB": tools["lib"], "TLEF": tools["tlef"],
               "LEF": tools["lef"], "CORE_SIDE": f"{max(20.0, (res['area_um2'] / 0.30) ** 0.5):.2f}"}
        (d / "pnr.tcl").write_text((HERE / "flow" / "pnr.tcl").read_text())   # OpenROAD's `source` rejects paths with spaces
        rc = run([tools["openroad"], "-no_init", "-exit", "pnr.tcl"], d, d / "pnr_report.txt", env)
        if rc != 0 or "=== REPORT_END" not in (d / "pnr_report.txt").read_text():
            res["pnr_status"] = "failed"
            res["pnr_detail"] = (d / "pnr_report.txt").read_text()[-600:]
        else:
            res["pnr_status"] = "ok"
            res.update(parse_pnr(d / "pnr_report.txt", period))
    res["seconds"] = round(time.time() - t0, 2)
    (d / "result.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--period", type=float, default=5.0, help="clock period in ns used for slack")
    ap.add_argument("--no-pnr", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    args.out = args.out.resolve()   # tools run with cwd = the design dir, so every path must be absolute
    tools = tool_paths()
    items = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    versions = {}
    for k in ("yosys", "openroad"):
        r = subprocess.run([tools[k], "-V" if k == "yosys" else "-version"], capture_output=True, text=True)
        versions[k] = (r.stdout + r.stderr).strip().splitlines()[0]
    (args.out / "run_meta.json").write_text(json.dumps({
        "argv": sys.argv, "period_ns": args.period, "pnr": not args.no_pnr, "n_designs": len(items),
        "tools": tools, "versions": versions, "pdk": str(PDK)}, indent=1))
    with ThreadPoolExecutor(args.jobs) as ex:
        results = list(ex.map(lambda it: process(it, args.out, tools, args.period, not args.no_pnr), items))
    with (args.out / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    bad = [r["design_id"] for r in results if r["status"] != "ok" or r.get("pnr_status") == "failed"]
    print(f"{len(results)} designs, {len(bad)} with a failed step: {bad}")


if __name__ == "__main__":
    main()
