"""Re-export the SiliconLM utilities we reuse (third_party/SiliconLM submodule), so nothing is copied."""

import sys
from pathlib import Path

_SILICONLM = Path(__file__).resolve().parents[1] / "third_party" / "SiliconLM"
if str(_SILICONLM) not in sys.path:
    sys.path.insert(0, str(_SILICONLM))

from silicon_eval.harness import PASS_TOKEN, extract_verilog, pass_at_k  # noqa: E402,F401
from data.build_corpus import near_dup, shingles  # noqa: E402,F401
