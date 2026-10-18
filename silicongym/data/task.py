"""Task record + JSONL (de)serialisation. A task = spec shown to the model + hidden testbench."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Task:
    id: str
    family: str
    spec: str                 # natural-language prompt shown to the model (includes module header)
    module_name: str
    module_header: str
    testbench: str            # hidden from the model
    test_names: list[str]     # names the testbench prints as `TEST <name> PASS|FAIL`
    reference: str            # known-correct module (for self-tests; never shown)
    params: dict = field(default_factory=dict)
    split: str = ""


def save_tasks(tasks: list[Task], path: Path) -> None:
    Path(path).write_text("".join(json.dumps(asdict(t)) + "\n" for t in tasks))


def load_tasks(path: Path) -> list[Task]:
    return [Task(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
