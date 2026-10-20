"""Exploratory (not preregistered): what D55's Cause swaps on max put out.

Max's predictions coincide: a sum's Cause target, max(S1', S2) for S1, is the source's own answer Y' whenever the
changed sum wins. A subspace carrying the whole answer then scores Cause on those pairs (Isolate catches it, as D55
says). Per model, read point and variable, at the fit's k*: how often the target is Y', and what the swap put out
(the target, Y', the base answer), overall and on the pairs whose target differs from Y'. The test split is rebuilt
from the config (it is a function of the seed); the stored per-base Cause hits check the rebuild.

    uv run python scripts/d55_decompose.py --runs <das_d55_* or das_d56_* run dirs> --out docs/das/d55_decompose.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import directions.das as das
from directions.config import load_das_config

TID0 = 15  # Qwen3: the digits 0-9 are the tokens 15-24 (checked against the stored hits below)


def decompose(run: Path, config: Path) -> dict:
    cfg = load_das_config(str(config))
    r = json.loads((run / "core" / "das.json").read_text())
    pred = dict(np.load(run / "core" / "das_predictions.npz"))
    cause_of = {v: das.cause_kind(cfg.task, v) for v in cfg.variables}
    causes = tuple(dict.fromkeys(cause_of.values()))
    kinds = {"train": causes, "val": causes,
             "test": tuple(k for k in das.TASK_SOURCE_KINDS[cfg.task] if not k.startswith("cause") or k in causes)}
    splits, _ = das.sample_splits(cfg.seed, {"train": cfg.n_train, "val": cfg.n_val, "test": cfg.n_test}, cfg.n_shots,
                                  cfg.n_demo_pool, kinds, None, cfg.task)
    te = splits["test"]
    out = {}
    for s in r["sites"]:
        for v in r["variables"]:
            k = r["cells"][f"{s}/{v}"]["k_star"]
            hits = pred[f"cause_hit__{s}/{v}/{k}"]
            rows = hits[:, 0]
            am = dict(pred[f"cause__{s}/{v}/{k}"].tolist())
            srcs = te.sources[cause_of[v]]
            t = np.array([das.targets(v, te.bases[i], srcs[i]) for i in rows])
            ysrc = np.array([srcs[i].Y for i in rows])
            y = np.array([te.bases[i].Y for i in rows])
            o = np.array([am[i] - TID0 for i in rows])
            if not (hits[:, 1] == (o == t)).all():
                raise RuntimeError(f"{run.name} {s}/{v}: the rebuilt split does not reproduce the stored Cause hits")
            diff = t != ysrc

            def rate(x: np.ndarray, m: np.ndarray | None = None) -> float | None:
                x = x if m is None else x[m]
                return round(float(x.mean()), 3) if len(x) else None

            out[f"{s}/{v}"] = {"n": int(len(rows)), "k_star": k, "target_is_source_answer": rate(~diff),
                               "out": {"target": rate(o == t), "source_answer": rate(o == ysrc), "base": rate(o == y)},
                               "n_target_differs": int(diff.sum()),
                               "out_where_target_differs": {"target": rate(o == t, diff), "source_answer": rate(o == ysrc, diff),
                                                            "base": rate(o == y, diff)}}
    return {"control_site": r["control_site"], "cells": out}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--configs", type=Path, default=Path("configs"), help="where das_d55_qwen3_<size>.yaml is, for a run "
                    "whose metadata names no config file")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    summary = {}
    for run in args.runs:
        size = re.search(r"qwen3_(\d+(?:\.\d+)?b)", run.name).group(1)
        meta = json.loads((run / "metadata.json").read_text())  # the run's own config (D55, D56, ...)
        config = Path(meta["config_path"]) if meta.get("config_path") else args.configs / f"das_d55_qwen3_{size}.yaml"
        res = decompose(run, config)
        summary[f"qwen3_{size}"] = res
        print(f"qwen3_{size} (control {res['control_site']}): target = source answer / out target, source answer, base"
              f" / where they differ (n): target, source answer")
        for key, c in res["cells"].items():
            w = c["out_where_target_differs"]
            print(f"  {key:5s} {c['target_is_source_answer']:.2f} / {c['out']['target']:.2f} {c['out']['source_answer']:.2f}"
                  f" {c['out']['base']:.2f} / ({c['n_target_differs']}) "
                  + (f"{w['target']:.2f} {w['source_answer']:.2f}" if w["target"] is not None else "-"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
