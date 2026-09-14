"""Which data centers currently have GPUs of each `run-gpu` tier (RunPod's availability signal).

    uv run python scripts/runpod_availability.py                    # every tier, every data center with stock
    uv run python scripts/runpod_availability.py --tier ADA_80_PRO  # one tier
    uv run python scripts/runpod_availability.py --datacenter EUR-NO-1
    uv run python scripts/runpod_availability.py --all             # also the data centers Flash cannot deploy to

Reads ``RUNPOD_API_KEY`` (a read-only key suffices). The signal is RunPod's per-data-center GPU
availability (``dataCenters.gpuAvailability`` in the GraphQL API, the same one the console shows, with
stock "Low"/"Medium"/"High"); it is the best proxy there is for whether a serverless job on that tier will
get a worker (docs/INFRA.md). Tier membership follows ``runpod_flash.GpuGroup``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

GRAPHQL_URL = "https://api.runpod.io/graphql"

# runpod_flash.core.resources.gpu.GpuGroup (1.19): tier -> RunPod GPU type ids
TIERS: dict[str, tuple[str, ...]] = {
    "AMPERE_16": ("NVIDIA RTX A4000", "NVIDIA RTX A4500", "NVIDIA RTX 4000 Ada Generation", "NVIDIA RTX 2000 Ada Generation"),
    "AMPERE_24": ("NVIDIA RTX A5000", "NVIDIA L4", "NVIDIA GeForce RTX 3090"),
    "ADA_24": ("NVIDIA GeForce RTX 4090",),
    "ADA_32_PRO": ("NVIDIA GeForce RTX 5090",),
    "AMPERE_48": ("NVIDIA A40", "NVIDIA RTX A6000"),
    "ADA_48_PRO": ("NVIDIA RTX 6000 Ada Generation", "NVIDIA L40", "NVIDIA L40S"),
    "AMPERE_80": ("NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"),
    "ADA_80_PRO": ("NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL"),
    "BLACKWELL_96": ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                     "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"),
    "HOPPER_141": ("NVIDIA H200",),
    "BLACKWELL_180": ("NVIDIA B200",),
}

# runpod_flash.core.resources.datacenter.DataCenter (1.19): the only data centers a Flash network volume
# (hence an endpoint of this workflow) can be placed in
FLASH_DATACENTERS = ("US-CA-2", "US-IL-1", "US-KS-2", "US-MO-1", "US-MO-2", "US-NC-2", "US-NE-1", "US-WA-1",
                     "EU-CZ-1", "EU-RO-1", "EUR-NO-1")

QUERY = "{ dataCenters { id listed gpuAvailability { gpuTypeId available stockStatus } } }"


def fetch(api_key: str) -> list[dict]:
    req = urllib.request.Request(
        GRAPHQL_URL, data=json.dumps({"query": QUERY}).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}", "User-Agent": "directions-runpod-availability"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.load(resp)
    if body.get("errors"):
        raise RuntimeError("GraphQL errors: " + "; ".join(e.get("message", str(e)) for e in body["errors"]))
    return body["data"]["dataCenters"]


def availability(datacenters: list[dict]) -> dict[str, dict[str, list[tuple[str, str | None]]]]:
    """``{tier: {datacenter: [(gpu type, stock), ...]}}`` over the GPU types reported available."""
    out: dict[str, dict[str, list[tuple[str, str | None]]]] = {t: {} for t in TIERS}
    for dc in datacenters:
        for g in dc.get("gpuAvailability") or []:
            if not g.get("available"):
                continue
            for tier, ids in TIERS.items():
                if g["gpuTypeId"] in ids:
                    out[tier].setdefault(dc["id"], []).append((g["gpuTypeId"].replace("NVIDIA ", ""), g.get("stockStatus")))
    return out


def format_table(table: dict[str, dict[str, list[tuple[str, str | None]]]], tier: str | None, datacenter: str | None,
                 flash_only: bool = True) -> str:
    """One line per tier: the data centers with stock (by default only those Flash can deploy to)."""
    lines = []
    for t, dcs in table.items():
        if tier and t != tier:
            continue
        rows = {d: v for d, v in dcs.items() if (not datacenter or d == datacenter) and (not flash_only or d in FLASH_DATACENTERS)}
        if not rows:
            lines.append(f"{t:<14} (none)")
            continue
        lines.append(f"{t:<14} " + "; ".join(f"{d}: " + ", ".join(f"{n} [{s}]" for n, s in v) for d, v in sorted(rows.items())))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", default=None, choices=sorted(TIERS))
    parser.add_argument("--datacenter", default=None)
    parser.add_argument("--all", action="store_true", help="every data center, not only the ones Flash can deploy to")
    args = parser.parse_args(argv)
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("RUNPOD_API_KEY is not set", file=sys.stderr)
        return 2
    print(format_table(availability(fetch(api_key)), args.tier, args.datacenter, flash_only=not args.all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
