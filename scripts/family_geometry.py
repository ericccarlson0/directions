"""The within-family geometry of a parameterised task family (docs/DECISIONS.md D39, reading iii).

    uv run python scripts/family_geometry.py --learned-run <run> [--fv-run <run>] \
        --labels kth_1,kth_2,kth_3,kth_4,kth_5 --params 1,2,3,4,5 [--name qwen3_0.6b] [--out geometry.json]

Reads the learned vectors (every candidate layer, with the three seed fits behind each) and the head-mean
vectors (at their selected layers) from finished run directories and prints, per layer, the cross-label cosines
against the seed spread and the ordered test (cosine falling with the parameter distance; exact label
permutation null). CPU only; nothing is recomputed on the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from directions.family import family_geometry, format_geometry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--learned-run", required=True, help="finished run with extraction.control: learned_vector")
    ap.add_argument("--fv-run", default=None, help="finished run with extraction.control: function_vector (optional)")
    ap.add_argument("--labels", default="kth_1,kth_2,kth_3,kth_4,kth_5", help="task keys, comma-separated, in parameter order")
    ap.add_argument("--params", default="1,2,3,4,5", help="the labels' parameter values, comma-separated")
    ap.add_argument("--name", default="", help="a name for the printout")
    ap.add_argument("--out", type=Path, default=None, help="write the geometry as JSON")
    args = ap.parse_args(argv)
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    params = [float(s) for s in args.params.split(",") if s.strip()]
    if len(labels) != len(params):
        ap.error("--labels and --params must have the same length")
    g = family_geometry(args.learned_run, labels, params, fv_run=args.fv_run)
    print(format_geometry(g, args.name))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(g, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
