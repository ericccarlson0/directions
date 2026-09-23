"""Run several stages of one model as one GPU job (docs/INFRA.md): each argument is one command line, run in
order as its own process (so the GPU is released between stages) with the job's environment already built by the
handler (``uv run --no-sync``; a command that does not start with ``uv run`` is run as given). The first failing
stage ends the job with its exit code; the stages before it have written their results and are kept.

usage: uv run python scripts/stages.py "uv run directions pilot --config ... --run-id A" \
           "uv run directions pilot --config ... --run-id B" \
           "uv run python scripts/landmarks.py --fv-run results/A --learned-run results/B --run-id C"
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time


def stage_argv(command: str) -> list[str]:
    """The process to run for one stage: ``uv run`` commands run in the job's environment without syncing it."""
    argv = shlex.split(command)
    if argv[:2] == ["uv", "run"] and "--no-sync" not in argv[2:3]:
        return ["uv", "run", "--no-sync", *argv[2:]]
    return argv


def main(commands: list[str]) -> int:
    if not commands:
        print("stages: no commands given", file=sys.stderr)
        return 2
    for i, command in enumerate(commands, 1):
        argv = stage_argv(command)
        print(f"[stages] {i}/{len(commands)} {shlex.join(argv)}", flush=True)
        t0 = time.time()
        code = subprocess.call(argv)
        print(f"[stages] {i}/{len(commands)} exit {code} after {time.time() - t0:.0f}s", flush=True)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
