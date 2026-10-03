"""The D44 readings across checkpoints, from downloaded runs (no model is loaded).

Per model, from the weight-decayed (D43) landmark comparison and source runs: the hand-over read point of the D43
vector and of the head mean per task (the first read point keeping 0.9 of the effect along the per-prompt natural
difference), the universal heads' write depth (D37's landmark), and attention's share of the steered and natural
aligning writes over the window to the hand-over with the removal readings (D38); beside them, where given, the
same numbers from the D31 landmark and source runs. From the mixing runs (D44's within-label path and D40's
cross-label path, on the D31 and D43 learned vectors): per label and other start, the within-label reading, and
per pair the cross-label verdict.

    uv run python scripts/d44_summary.py --model qwen3_0.6b --wd-landmarks <run> --wd-source <run> \\
        [--d31-landmarks <run>] [--d31-source <run>] --wd-mixing <run> --d31-mixing <run> [...more --model groups] \\
        --out results/d44_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any


def _json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def handover(run: Path | None) -> dict[str, Any] | None:
    """Per task: the hand-over read point of the learned vector and the head mean, the primary layer; and the
    model's landmarks (the universal heads' median write read point)."""
    if run is None:
        return None
    lm = _json(run / "landmarks" / "landmarks.json")
    if lm is None:
        return None
    tasks = {}
    for name, t in (lm.get("tasks") or {}).items():
        h = t.get("handover") or {}
        tasks[name] = {"layer": t.get("primary_layer"),
                       "learned": (h.get("learned") or {}).get("read_point"), "fv": (h.get("fv") or {}).get("read_point")}
    summ = lm.get("summary") or {}
    return {"n_layers": lm.get("n_layers"), "tasks": tasks, "median": summ.get("median"),
            "offset_from_handover": summ.get("offset_from_handover")}


def source(run: Path | None) -> dict[str, Any] | None:
    """Per task: attention's share of the aligning write over the window to the hand-over (steered, natural,
    random), and the effect retained when the aligning components of each sublayer are removed (with the matched
    random removal)."""
    if run is None:
        return None
    s = _json(run / "core" / "summary.json")
    if s is None:
        return None
    out = {}
    for name, t in (s.get("tasks") or {}).items():
        w = (t.get("windows") or {}).get("to_handover") or {}
        share = {k: (w.get(k) or {}).get("att_share", {}).get("median") for k in ("steered", "natural", "random")}
        verdict = (w.get("steered") or {}).get("verdict")
        rem = t.get("removal") or {}
        removal = {k: {"retained": (rem.get(k) or {}).get("retained"), "random": (rem.get(k) or {}).get("random_retained")}
                   for k in ("att_along", "mlp_along")}
        out[name] = {"handover": t.get("handover_read_point"), "att_share": share, "steered_verdict": verdict,
                     "removal": removal}
    return out


def mixing(run: Path | None) -> dict[str, Any] | None:
    """The within-label readings per label and start, and the cross-label verdicts per pair."""
    if run is None:
        return None
    s = _json(run / "core" / "summary.json")
    if s is None:
        return None
    within: dict[str, Any] = {}
    cross: list[dict[str, Any]] = []
    for p in s.get("pairs") or []:
        if p.get("construction") == "within":
            for label, v in (p.get("per_label") or {}).items():
                for path in v.get("paths") or []:
                    within[f"{p['family']}/{label}/s{path['seed']}"] = {
                        "reading": path["reading"], "cos": path["cos"], "retained_mid": path["retained_mid"], "p_mid": path["p_mid"]}
        elif p.get("construction") == "learned":
            cross.append({"family": p["family"], "labels": p["labels"], "verdict": p.get("verdict"), "angle_deg": p.get("angle_deg")})
    return {"within": within, "cross": cross, "skipped": s.get("skipped")}


def _med(xs: list[Any]) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(median(xs)) if xs else None


def summarise(models: dict[str, dict[str, Path | None]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model, runs in models.items():
        r = {k: f(runs.get(k)) for k, f in (("wd_landmarks", handover), ("d31_landmarks", handover), ("wd_source", source),
                                              ("d31_source", source), ("wd_mixing", mixing), ("d31_mixing", mixing))}
        row: dict[str, Any] = {"runs": {k: str(v) if v else None for k, v in runs.items()}, **r}
        wl, dl = r["wd_landmarks"], r["d31_landmarks"]
        if wl:
            row["handover_median"] = {"wd": (wl.get("median") or {}).get("handover_learned"),
                                      "fv": (wl.get("median") or {}).get("handover_fv"),
                                      "heads": (wl.get("median") or {}).get("heads_median"),
                                      "d31": (dl.get("median") or {}).get("handover_learned") if dl else None,
                                      "heads_offset_wd": ((wl.get("offset_from_handover") or {}).get("heads_median") or {}).get("median")}
        for kind in ("wd", "d31"):
            src = r[f"{kind}_source"]
            if src:
                row[f"{kind}_att_share_steered_median"] = _med([t["att_share"]["steered"] for t in src.values()])
                row[f"{kind}_att_share_natural_median"] = _med([t["att_share"]["natural"] for t in src.values()])
            mx = r[f"{kind}_mixing"]
            if mx:
                counts: dict[str, int] = {}
                for v in mx["within"].values():
                    counts[v["reading"]] = counts.get(v["reading"], 0) + 1
                row[f"{kind}_within_counts"] = counts
        out[model] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True)
    for k in ("wd-landmarks", "d31-landmarks", "wd-source", "d31-source", "wd-mixing", "d31-mixing"):
        ap.add_argument(f"--{k}", action="append", default=[])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    n = len(args.model)
    models: dict[str, dict[str, Path | None]] = {}
    for i, m in enumerate(args.model):
        models[m] = {}
        for k in ("wd_landmarks", "d31_landmarks", "wd_source", "d31_source", "wd_mixing", "d31_mixing"):
            vals = getattr(args, k)
            if vals and len(vals) != n:
                raise SystemExit(f"--{k.replace('_', '-')} must be given once per --model (use 'none')")
            v = vals[i] if vals else None
            models[m][k] = None if v in (None, "none") else Path(v)
    res = summarise(models)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1, default=float))
    for m, row in res.items():
        print(f"== {m}: hand-over medians {row.get('handover_median')}; attention share (steered, natural) "
              f"wd {row.get('wd_att_share_steered_median')}, {row.get('wd_att_share_natural_median')}; "
              f"d31 {row.get('d31_att_share_steered_median')}, {row.get('d31_att_share_natural_median')}; "
              f"within wd {row.get('wd_within_counts')}, d31 {row.get('d31_within_counts')}")


if __name__ == "__main__":
    main()
