"""The RunPod account balance, and whether it covers a planned run (docs/INFRA.md).

    uv run python scripts/runpod_balance.py                 # balance, spend limit, current spend rate
    uv run python scripts/runpod_balance.py --need 6        # exit 1 unless the balance covers 6 USD (plus a margin)
    uv run python scripts/runpod_balance.py --hours 1.5 --rate 4.79   # need = hours * rate

Reads ``RUNPOD_API_KEY`` (a read-only key suffices; ``myself.clientBalance`` in the GraphQL API). Run it before
requesting a long job: a job submitted with the credit exhausted is refused with HTTP 402 at submission, and one
that exhausts the credit mid-run is killed without results. The gate adds a margin (``--margin``, default 50 %)
on top of the estimate, since queue time and retries are billed too.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

GRAPHQL_URL = "https://api.runpod.io/graphql"
QUERY = "{ myself { clientBalance spendLimit currentSpendPerHr } }"


def fetch(api_key: str) -> dict:
    req = urllib.request.Request(
        GRAPHQL_URL, data=json.dumps({"query": QUERY}).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}", "User-Agent": "directions-runpod-balance"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.load(resp)
    if body.get("errors"):
        raise RuntimeError("GraphQL errors: " + "; ".join(e.get("message", str(e)) for e in body["errors"]))
    return body["data"]["myself"]


def required(need: float | None, hours: float | None, rate: float | None, margin: float) -> float | None:
    """The balance a planned run needs: ``need`` USD, or ``hours * rate``, times ``1 + margin``; None without a plan."""
    if need is None and (hours is None or rate is None):
        return None
    estimate = need if need is not None else hours * rate
    return estimate * (1.0 + margin)


def verdict(balance: float, needed: float | None) -> tuple[bool, str]:
    """(ok, one line): whether ``balance`` covers ``needed`` (always ok without a plan)."""
    if needed is None:
        return True, f"balance {balance:.2f} USD"
    ok = balance >= needed
    return ok, f"balance {balance:.2f} USD, needed {needed:.2f} USD (estimate with margin): {'ok' if ok else 'INSUFFICIENT'}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--need", type=float, default=None, help="planned cost in USD")
    parser.add_argument("--hours", type=float, default=None, help="planned wall-clock hours (with --rate)")
    parser.add_argument("--rate", type=float, default=None, help="USD per hour of the tier (with --hours)")
    parser.add_argument("--margin", type=float, default=0.5, help="safety margin on the estimate (fraction, default 0.5)")
    args = parser.parse_args(argv)
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("RUNPOD_API_KEY is not set", file=sys.stderr)
        return 2
    me = fetch(api_key)
    balance = float(me.get("clientBalance") or 0.0)
    ok, line = verdict(balance, required(args.need, args.hours, args.rate, args.margin))
    print(line)
    print(f"spend limit {me.get('spendLimit')} USD, current spend {me.get('currentSpendPerHr')} USD/h")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
