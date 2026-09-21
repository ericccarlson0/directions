"""Free space on the RunPod network volume from a GPU job (docs/INFRA.md): report what occupies it and delete
named Hugging Face model caches and incomplete downloads. Runs as the request's command (`uv run python
scripts/volume_cleanup.py ...`), where the volume is mounted at /runpod-volume and HF_HOME is /runpod-volume/hf.

Dry run unless --yes: the report is printed either way.

usage: uv run python scripts/volume_cleanup.py [--delete-model google/gemma-4-12B ...] [--delete-incomplete] [--yes]
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

VOLUME = Path(os.environ.get("DIRECTIONS_VOLUME_ROOT", "/runpod-volume"))


def du(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda e: None):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def gb(n: int) -> str:
    return f"{n / 2**30:6.2f} GB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete-model", action="append", default=[], help="HF repo id whose cache directory is deleted")
    ap.add_argument("--delete-incomplete", action="store_true", help="delete *.incomplete blobs (partial downloads)")
    ap.add_argument("--yes", action="store_true", help="actually delete (otherwise a dry run)")
    args = ap.parse_args()

    if not VOLUME.is_dir():
        raise SystemExit(f"{VOLUME} is not mounted")
    st = shutil.disk_usage(VOLUME)
    print(f"volume {VOLUME}: total {gb(st.total)} used {gb(st.used)} free {gb(st.free)}", flush=True)
    for top in sorted(VOLUME.iterdir()):
        if top.is_dir():
            print(f"  {gb(du(top))}  {top.name}/", flush=True)
    hub = VOLUME / "hf" / "hub"
    if hub.is_dir():
        for d in sorted(hub.iterdir()):
            if d.is_dir() and d.name.startswith("models--"):
                inc = sum(1 for _ in d.rglob("*.incomplete"))
                print(f"  {gb(du(d))}  hf/hub/{d.name}/  ({inc} incomplete blobs)", flush=True)
    results = VOLUME / "results"
    if results.is_dir():
        for d in sorted(results.iterdir()):
            print(f"  {gb(du(d))}  results/{d.name}/", flush=True)

    targets: list[Path] = []
    for repo in args.delete_model:
        d = hub / ("models--" + repo.replace("/", "--"))
        if d.is_dir():
            targets.append(d)
        else:
            print(f"no cache directory for {repo} ({d})", flush=True)
    if args.delete_incomplete and hub.is_dir():
        targets.extend(hub.rglob("*.incomplete"))
    for t in targets:
        size = du(t) if t.is_dir() else t.stat().st_size
        print(f"{'deleting' if args.yes else 'would delete'} {gb(size)} {t}", flush=True)
        if args.yes:
            if t.is_dir():
                shutil.rmtree(t, ignore_errors=True)
            else:
                t.unlink(missing_ok=True)
    if args.yes and targets:
        st = shutil.disk_usage(VOLUME)
        print(f"after: used {gb(st.used)} free {gb(st.free)}", flush=True)


if __name__ == "__main__":
    main()
