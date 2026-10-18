"""Tables and plots for analysis/report.ipynb. Every function reads logs under runs/ and returns
markdown (or draws a figure); with no logs it says so instead of inventing anything."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from silicongym.eval.tables import table

NONE = "_no data found under {path} -- run the corresponding phase first._"


def eval_table(eval_dir: str | Path, split: str = "test", name_filter=None) -> str:
    out = table(Path(eval_dir), split, name_filter)
    return out if out.count("\n") > 1 else NONE.format(path=eval_dir)


def hack_by_type(eval_dir: str | Path, split: str = "test") -> str:
    """Mean per-type hack counts per model (summed over samples), from eval JSONs."""
    runs: dict[str, list[dict]] = {}
    for p in sorted(Path(eval_dir).glob(f"*_{split}_seed*.json")):
        r = json.loads(p.read_text())
        runs.setdefault(r["name"], []).append(r["metrics"]["hack"])
    if not runs:
        return NONE.format(path=eval_dir)
    types = list(next(iter(runs.values()))[0]["by_type"])
    lines = ["| model | seeds | hack rate | " + " | ".join(types) + " |", "|---|---|---|" + "---|" * len(types)]
    for name, hs in runs.items():
        cells = [f"{statistics.mean(h['by_type'][t] for h in hs):.1f}" for t in types]
        lines.append(f"| {name} | {len(hs)} | {statistics.mean(h['hack_rate'] for h in hs):.3f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def hack_examples_md(path: str | Path, max_chars: int = 600) -> str:
    p = Path(path)
    if not p.exists():
        return NONE.format(path=path)
    out = []
    for kind, items in json.loads(p.read_text()).items():
        out.append(f"### {kind} ({len(items)} saved)")
        for it in items:
            out.append(f"- run `{it['run']}`, task `{it['task_id']}`, reward {it['reward']}, {it['detail']}\n"
                       f"```verilog\n{it['code'][:max_chars]}\n```")
        if not items:
            out.append("_none observed_")
    return "\n".join(out)


def load_curves(runs_dir: str | Path) -> dict[str, list[dict]]:
    """run name -> list of per-step metric dicts, from metrics.jsonl (custom loops) or samples.jsonl (TRL)."""
    curves: dict[str, list[dict]] = {}
    for d in sorted(Path(runs_dir).glob("*_seed*")):
        m, s = d / "metrics.jsonl", d / "samples.jsonl"
        if m.exists():
            curves[d.name] = [json.loads(line) for line in m.read_text().splitlines() if line.strip()]
        elif s.exists():          # TRL path: one reward-fn call per generation batch
            by_call: dict[int, list[dict]] = {}
            for line in s.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    by_call.setdefault(row["call"], []).append(row)
            curves[d.name] = [{
                "step": c, "reward": statistics.mean(r["reward"] for r in rows),
                "all_passed": statistics.mean(bool(r["all_passed"]) for r in rows),
                "hack_rate": statistics.mean(bool(r["hacked"]) for r in rows)} for c, rows in sorted(by_call.items())]
    return curves


def plot_curves(runs_dir: str | Path, keys=("reward", "all_passed", "hack_rate")):
    import matplotlib.pyplot as plt
    curves = load_curves(runs_dir)
    if not curves:
        print(NONE.format(path=runs_dir))
        return None
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.5))
    for ax, key in zip(axes, keys):
        for name, rows in curves.items():
            ax.plot([r["step"] for r in rows], [r[key] for r in rows], label=name, linewidth=1)
        ax.set_title(key); ax.set_xlabel("step")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    return fig


def agreement_md(path: str | Path = "runs/agreement.json") -> str:
    p = Path(path)
    if not p.exists():
        return NONE.format(path=path)
    rep = json.loads(p.read_text())
    f = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    lines = [f"{rep['n']} samples, {rep['n_ground_truth_pass']} pass the strict simulator check.", "",
             "| reward | AUROC | Spearman vs pass rate | accuracy | false-accept | false-reject |", "|---|---|---|---|---|---|"]
    for name, m in rep["scorers"].items():
        lines.append(f"| {name} | {f(m['auroc'])} | {f(m['spearman_vs_pass_rate'])} | {f(m['accuracy'])} | "
                     f"{f(m['false_accept_rate'])} | {f(m['false_reject_rate'])} |")
    for name, m in rep["scorers"].items():
        for kind in ("false_accepts", "false_rejects"):
            for ex in m[kind][:1]:
                lines += ["", f"**{name} {kind[:-1].replace('_', ' ')}** (score {ex['score']:.2f}, true pass rate "
                          f"{ex['gt_rate']:.2f}, task `{ex['task_id']}`)", f"```verilog\n{ex['code'][:500]}\n```"]
    return "\n".join(lines)


def regression_md(reg_dir: str | Path = "runs/regression") -> str:
    rows = [r for r in (json.loads(p.read_text()) for p in sorted(Path(reg_dir).glob("*.json"))) if "model" in r]
    if not rows:
        return NONE.format(path=reg_dir)
    lines = ["| model | HumanEval-subset pass@1 | base pass@1 | delta |", "|---|---|---|---|"]
    for r in rows:
        base = r.get("base", {}).get("pass@1")
        base_s = "n/a" if base is None else f"{base:.3f}"
        delta_s = "n/a" if base is None else "{:+.3f}".format(r["delta_pass@1"])
        lines.append(f"| {r['name']} | {r['model']['pass@1']:.3f} | {base_s} | {delta_s} |")
    return "\n".join(lines)
