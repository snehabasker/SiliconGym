"""Learned reward model trained on preference pairs {"prompt","chosen","rejected"}.

The pair format is the one SiliconLM's `make_dpo_pairs.py` writes, so its existing DPO pairs work as-is.
`make-pairs` builds the same format from scored samples (passing vs failing code for the same task).

python -m silicongym.rewards.learned_rm make-pairs --samples runs/eval/*_train_*.samples.jsonl --out pairs.jsonl
python -m silicongym.rewards.learned_rm train --pairs pairs.jsonl --out runs/rm --seed 0

Model: Qwen2.5-Coder + LoRA + a scalar head (AutoModelForSequenceClassification), Bradley-Terry loss
-log sigmoid(r_chosen - r_rejected). Train it on train-split pairs only: agreement is measured on test.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

from silicongym.data.task import load_tasks

MAX_LEN = 768


def rm_text(spec: str, code: str) -> str:
    return f"Specification:\n{spec}\n\nVerilog:\n{code}"


def load_pairs(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def make_pairs(samples: list[dict], tasks: dict, max_per_task: int = 8, seed: int = 0) -> list[dict]:
    """Pair every passing code with a failing one for the same task (distinct code only)."""
    rng = random.Random(seed)
    by_task: dict[str, dict[str, list[str]]] = {}
    for s in samples:
        if not s.get("code") or s["task_id"] not in tasks:
            continue
        by_task.setdefault(s["task_id"], {"good": [], "bad": []})["good" if s["all_passed"] else "bad"].append(s["code"])
    pairs = []
    for tid, d in by_task.items():
        combos = list(itertools.product(dict.fromkeys(d["good"]), dict.fromkeys(d["bad"])))
        rng.shuffle(combos)
        pairs += [{"prompt": tasks[tid].spec, "chosen": c, "rejected": r} for c, r in combos[:max_per_task]]
    return pairs


class RewardModel:
    def __init__(self, tok, model):
        self.tok, self.model = tok, model

    @classmethod
    def load(cls, path: str, base: str | None = None, load_4bit: bool = True) -> "RewardModel":
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

        from silicongym.train.model import DEFAULT_BASE
        base = base or DEFAULT_BASE
        tok = AutoTokenizer.from_pretrained(base)
        tok.pad_token = tok.pad_token or tok.eos_token
        kw: dict = {"num_labels": 1}
        if load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        if torch.cuda.is_available():
            kw["device_map"] = {"": 0}
        model = AutoModelForSequenceClassification.from_pretrained(base, **kw)
        model.config.pad_token_id = tok.pad_token_id
        model = PeftModel.from_pretrained(model, path)
        return cls(tok, model.eval())

    def score(self, spec: str, code: str) -> float:
        """Reward logit; higher means more likely to be the 'chosen' (passing) design."""
        import torch
        enc = self.tok(rm_text(spec, code), return_tensors="pt", truncation=True, max_length=MAX_LEN)
        with torch.no_grad():
            return float(self.model(**enc.to(self.model.device)).logits[0, 0])


def train_rm(pairs: list[dict], out: Path, base: str, seed: int, epochs: int = 2, lr: float = 1e-4,
             batch_size: int = 4, load_4bit: bool = True, val_frac: float = 0.1) -> dict:
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, set_seed

    set_seed(seed)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_frac))
    val, train = pairs[:n_val], pairs[n_val:]

    tok = AutoTokenizer.from_pretrained(base)
    tok.pad_token = tok.pad_token or tok.eos_token
    kw: dict = {"num_labels": 1}
    if load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    if torch.cuda.is_available():
        kw["device_map"] = {"": 0}
    model = AutoModelForSequenceClassification.from_pretrained(base, **kw)
    model.config.pad_token_id = tok.pad_token_id
    if load_4bit:
        model = prepare_model_for_kbit_training(model)
    from silicongym.train.model import LORA_TARGETS
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
                                             target_modules=LORA_TARGETS, modules_to_save=["score"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    def rewards(spec_code: list[tuple[str, str]]):
        enc = tok([rm_text(s, c) for s, c in spec_code], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(model.device)
        return model(**enc).logits[:, 0].float()

    def pair_batch(batch):
        r = rewards([(p["prompt"], p["chosen"]) for p in batch] + [(p["prompt"], p["rejected"]) for p in batch])
        return r[: len(batch)], r[len(batch):]

    def val_acc():
        model.eval()
        with torch.no_grad():
            wins = [float(c > r) for i in range(0, len(val), batch_size) for c, r in zip(*pair_batch(val[i:i + batch_size]))]
        model.train()
        return sum(wins) / len(wins)

    log = []
    model.train()
    for epoch in range(epochs):
        rng.shuffle(train)
        for i in range(0, len(train), batch_size):
            c, r = pair_batch(train[i:i + batch_size])
            loss = -F.logsigmoid(c - r).mean()
            loss.backward()
            opt.step(); opt.zero_grad()
            log.append(float(loss.detach()))
        print(f"epoch {epoch}: loss {sum(log[-10:]) / len(log[-10:]):.4f} val pair accuracy {val_acc():.3f}")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    stats = {"n_train_pairs": len(train), "n_val_pairs": len(val), "val_pair_accuracy": val_acc(),
             "final_loss": sum(log[-10:]) / len(log[-10:]), "seed": seed, "base": base,
             "epochs": epochs, "lr": lr}
    (out / "rm_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make-pairs")
    mk.add_argument("--samples", nargs="+", type=Path, required=True)
    mk.add_argument("--tasks", type=Path, default=Path("silicongym/data/tasks/train.jsonl"))
    mk.add_argument("--out", type=Path, required=True)
    mk.add_argument("--max-per-task", type=int, default=8)
    tr = sub.add_parser("train")
    tr.add_argument("--pairs", type=Path, required=True)
    tr.add_argument("--out", type=Path, required=True)
    tr.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--epochs", type=int, default=2)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--no-4bit", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "make-pairs":
        samples = [json.loads(line) for p in a.samples for line in p.read_text().splitlines() if line.strip()]
        pairs = make_pairs(samples, {t.id: t for t in load_tasks(a.tasks)}, a.max_per_task)
        a.out.write_text("".join(json.dumps(p) + "\n" for p in pairs))
        print(f"wrote {len(pairs)} pairs to {a.out}")
    else:
        print(json.dumps(train_rm(load_pairs(a.pairs), a.out, a.base, a.seed, a.epochs, a.lr,
                                  load_4bit=not a.no_4bit), indent=2))


if __name__ == "__main__":
    main()
