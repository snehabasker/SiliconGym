"""Run external tools in a throwaway directory with a timeout, scrubbed env and capped output.

Testbench protection (see Workspace.remove): the testbench is compiled in the workspace, then
deleted together with the DUT source before the simulator runs, so the running simulation has an
empty cwd. The compiled .vvp still embeds the testbench, so file/system/include constructs are
refused outright by `violations` before anything is compiled.

Network: iverilog/vvp load no network code unless a VPI module is passed with -m, which no
tool here does. On Linux we additionally try `unshare -n`; elsewhere that layer is absent.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

MAX_OUTPUT_BYTES = 64_000
DEFAULT_TIMEOUT_S = 10.0

# File access, shell escapes and includes: never needed for a synthesizable module.
_FORBIDDEN = re.compile(
    r"\$(?:fopen|fclose|fread|fscanf|fgets|fgetc|ungetc|fseek|ftell|rewind|readmem[bh]|writemem[bh]|"
    r"system|dumpfile|dumpvars|dumpon|dumpoff|dumpall|dumpflush|sdf_annotate|"
    r"value\$plusargs|test\$plusargs|fwrite|fdisplay|fstrobe|fmonitor)\b"
    r"|`\s*include\b"
    r"|\bvpi_|\bacc_"
)
_COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


_SIM_CONTROL = re.compile(r"\$(?:finish|stop|fatal|exit)\b")


def violations(code: str, ban_sim_control: bool = False) -> list[str]:
    """Forbidden constructs in `code` (comments ignored). Non-empty means refuse to run it.

    `ban_sim_control` also refuses $finish/$stop/$fatal/$exit. It is a reward-hacking safeguard,
    off by default so the naive baseline can exhibit the hack.
    """
    body = _COMMENTS.sub("", code)
    found = {m.group(0).strip() for m in _FORBIDDEN.finditer(body)}
    if ban_sim_control:
        found |= {m.group(0) for m in _SIM_CONTROL.finditer(body)}
    return sorted(found)


def find_tool(name: str) -> str | None:
    return shutil.which(name)


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def _limits():  # pragma: no cover - runs in the child on Linux only
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 << 20, 32 << 20))
    os.setsid()


_NETLESS_PREFIX: list[str] | None = None


def _no_network_prefix() -> list[str]:
    """['unshare', '-n'] if the kernel lets us, probed once; otherwise []."""
    global _NETLESS_PREFIX
    if _NETLESS_PREFIX is None:
        _NETLESS_PREFIX = []
        if os.name == "posix" and shutil.which("unshare"):
            probe = subprocess.run(["unshare", "-n", "true"], capture_output=True)
            if probe.returncode == 0:
                _NETLESS_PREFIX = ["unshare", "-n"]
    return _NETLESS_PREFIX


def _read_capped(path: Path) -> str:
    with path.open("rb") as f:
        return f.read(MAX_OUTPUT_BYTES).decode("utf-8", errors="replace")


class Workspace:
    """A temp directory that is the only place tools run; deleted on exit."""

    def __init__(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="sgym_"))
        self._out = Path(tempfile.mkdtemp(prefix="sgym_out_"))  # stdout/stderr live outside cwd

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)
        shutil.rmtree(self._out, ignore_errors=True)

    def write(self, name: str, text: str) -> Path:
        path = (self.dir / name).resolve()
        if self.dir.resolve() not in path.parents:
            raise ValueError(f"path escapes workspace: {name}")
        path.write_text(text, encoding="utf-8", errors="replace")  # model output can be any unicode
        return path

    def remove(self, *names: str) -> None:
        for n in names:
            (self.dir / n).unlink(missing_ok=True)

    def run(self, argv: list[str], timeout: float = DEFAULT_TIMEOUT_S) -> RunResult:
        """Run argv in the workspace. Timeouts and missing binaries come back as results, never raise."""
        if find_tool(argv[0]) is None:
            return RunResult(127, "", f"{argv[0]}: command not found")
        env = {k: v for k, v in os.environ.items()
               if k in {"PATH", "SYSTEMROOT", "TEMP", "TMP", "LD_LIBRARY_PATH"}}
        out_p, err_p = self._out / "stdout", self._out / "stderr"
        kwargs: dict = {}
        if os.name == "posix":
            kwargs["preexec_fn"] = _limits
        with out_p.open("wb") as out, err_p.open("wb") as err:
            proc = subprocess.Popen(_no_network_prefix() + argv, cwd=self.dir, env=env,
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=err, **kwargs)
            timed_out = False
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   capture_output=True)
                proc.wait()
        return RunResult(proc.returncode, _read_capped(out_p), _read_capped(err_p), timed_out)
