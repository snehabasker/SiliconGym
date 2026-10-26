# eda_study: open-source EDA on SiliconGym designs

**Status: work in progress.** Only the golden-design synthesis is finished. Nothing below compares model-generated
designs with golden ones yet; that code is written but has not been run.

## What is done

All 96 golden designs (`silicongym/data/tasks/{train,val,test}.jsonl`, field `reference`) were run through:

1. **Yosys** synthesis to `sky130_fd_sc_hd` (tt, 25 C, 1.8 V): `dfflibmap`, `abc -liberty`, `hilomap`, `insbuf`, `stat -liberty`.
2. **OpenROAD** placement-level flow (`flow/pnr.tcl`): floorplan, pin placement, global placement, `repair_design`,
   detailed placement, clock-tree synthesis (sequential designs only), placement-based parasitic estimate, then
   worst setup slack, critical path and power reports.

96 of 96 designs synthesised and went through the placement flow (32 sequential, 64 combinational). Summary of the raw
output in `results/golden/results.jsonl` (min / median / max over the 96 designs):

| metric | min | median | max |
|---|---|---|---|
| cells after synthesis | 1 | 8 | 278 |
| synthesised area (um^2) | 8.76 | 92.6 | 2129.5 |
| placed design area (um^2) | 9 | 128 | 2430 |
| worst setup slack at a 5 ns clock (ns) | 2.48 | 4.46 | 4.88 |
| total power (W) | 3.04e-07 | 9.31e-06 | 2.49e-04 |

Per-design logs, netlists, reports and the tool versions and paths are under `results/golden/`
(`run_meta.json`, `designs/*/`), and the console log is `results/golden_run.log`.

## Files

- `eda_run.py`: runs Yosys (+ optional formal equivalence to the golden design) and OpenROAD for a manifest of designs.
- `flow/pnr.tcl`: the OpenROAD script.
- `make_golden_manifest.py`: builds `manifests/golden.jsonl`.
- `collect_generated.py`, `analyze_eda.py`: select model-generated designs that pass every hidden test and compare them
  with the golden ones (per-task geometric-mean ratio, bootstrap interval, sign test). **Written, not run yet.**

## Not done yet

- Sampling the base model, synthesising the designs that passed, and the generated-vs-golden comparison.
- The area-aware reward experiment, which depends on that comparison showing generated designs are larger.

## Limitations

- This is **not OpenLane** and not a signoff flow. There is no power grid, no tap cells, no detailed routing and no
  DRC/LVS. "Area" is cell area after synthesis or after placement, not a laid-out die.
- OpenROAD is a March 2022 build (`f12e2f474`), the version available from conda for this setup.
- Timing uses a single 5 ns clock and zero I/O delays, with parasitics estimated from placement. Combinational designs
  use a virtual clock. Slack here says little about what a design could achieve at a tighter clock.
- Power uses OpenROAD's default switching activity, not activity from a testbench, so treat it as a rough comparison
  between designs, not an absolute figure.
- The designs are small (median 8 cells). The core is never smaller than 20 um on a side, so placed area of the
  smallest designs is dominated by that floor.
- No chip was fabricated or taped out.

## Environment

WSL2 Ubuntu, tools installed in user space with micromamba: Yosys 0.69+, Icarus Verilog 13.0, OpenROAD `f12e2f474`,
sky130A from open_pdks 1.0.471. Exact paths and versions are in `results/golden/run_meta.json`.

Reproduce: `python eda_run.py --manifest manifests/golden.jsonl --out results/golden --jobs 2` (tool paths default to
`~/eda/mamba/envs/{eda,orlh,pdk}`).
