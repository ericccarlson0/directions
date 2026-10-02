"""Compare the weight-decayed learned vector (docs/DECISIONS.md D42) with the D31 learned vector and the head-mean
function vector of the same model and seed, from three finished pilot runs (no model is loaded).

Per task: whether each construction qualified, its selected layer and strength, its held-out effect (mean d(lp/tok)
on the evaluation pool), its excess over the gate controls, its damage (query-token and neutral-prose KL); and per
candidate layer the geometry: the weight-decayed vector's fitted norm over the median residual norm, its cosine with
the D31 vector, the function vector and PC1, the D31 vector's cosine with the function vector, the cross-seed
stability of both learned constructions, the final fitting losses, and how much of a random start survives in the
weight-decayed fit. Writes a JSON file and prints a markdown table.

    uv run python scripts/learned_wd_compare.py --wd-run <learned_wd run> --learned-run <learned run> \\
        --fv-run <pilot run> --out <file.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _unit(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64) / np.linalg.norm(v)


def _held_out(q: dict[str, Any] | None) -> dict[str, Any]:
    """The selected condition's held-out numbers from a task's qualification.json (None when not measured)."""
    if not q or not q.get("selection") or not q.get("steering"):
        return {"qualified": bool(q and q.get("qualified")), "measured": False}
    s, sel = q["steering"], q["selection"]
    return {
        "qualified": bool(q.get("qualified")), "measured": True,
        "layer": sel["layer"], "rho": sel["rho"], "rho_layer_norm": sel.get("rho_layer_norm"), "alpha": sel["alpha"],
        "effect": s["test"]["mean_diff"], "effect_p": s["test"]["p_value"],
        "excess": s["excess_test"]["excess_mean"], "excess_p": s["excess_test"]["p_value"],
        "kl_mean": (q.get("damage") or {}).get("kl_mean"), "neutral_kl_mean": (q.get("damage") or {}).get("neutral_kl_mean"),
    }


def compare(wd_run: Path, learned_run: Path, fv_run: Path) -> dict[str, Any]:
    meta = {k: _json(r / "metadata.json") for k, r in (("wd", wd_run), ("learned", learned_run), ("fv", fv_run))}
    controls = {k: (m or {}).get("control") for k, m in meta.items()}
    expected = {"wd": "learned_vector_wd", "learned": "learned_vector", "fv": "function_vector"}
    if controls != expected:
        raise SystemExit(f"run controls {controls}, expected {expected}")
    seeds = {k: (m or {}).get("seed") for k, m in meta.items()}
    if len(set(seeds.values())) != 1:
        raise SystemExit(f"the runs have different seeds: {seeds}")
    out: dict[str, Any] = {"runs": {"wd": str(wd_run), "learned": str(learned_run), "fv": str(fv_run)}, "seed": seeds["wd"],
                           "model": (meta["wd"].get("model") or {}), "tasks": {}}
    for tdir in sorted((wd_run / "core" / "tasks").iterdir()):
        name = tdir.name
        row: dict[str, Any] = {}
        for k, root in (("wd", wd_run), ("learned", learned_run), ("fv", fv_run)):
            row[k] = _held_out(_json(root / "core" / "tasks" / name / "qualification.json"))
        a_wd = np.load(tdir / "directions.npz") if (tdir / "directions.npz").exists() else None
        lf = learned_run / "core" / "tasks" / name / "directions.npz"
        ff = fv_run / "core" / "tasks" / name / "directions.npz"
        a_l = np.load(lf) if lf.exists() else None
        a_f = np.load(ff) if ff.exists() else None
        ext_wd = (_json(tdir / "extraction.json") or {}).get("learned_vector") or {}
        ext_l = (_json(learned_run / "core" / "tasks" / name / "extraction.json") or {}).get("learned_vector") or {}
        geometry: dict[str, Any] = {}
        if a_wd is not None and "learned" in a_wd.files:
            layers = [int(l) for l in a_wd["layers"]]
            for i, l in enumerate(layers):
                g: dict[str, Any] = {}
                w = _unit(a_wd["learned"][i])
                g["fitted_norm_over_radius"] = float(a_wd["learned_radii"][i] / a_wd["learned_reference_radii"][i])
                g["cos_pca"] = float(w @ _unit(a_wd["pooled"][i]))
                if a_l is not None and "learned" in a_l.files and l in [int(x) for x in a_l["layers"]]:
                    j = [int(x) for x in a_l["layers"]].index(l)
                    d31 = _unit(a_l["learned"][j])
                    g["cos_d31"] = float(w @ d31)
                    if a_f is not None and "fv_direction" in a_f.files:
                        g["cos_d31_fv"] = float(d31 @ _unit(a_f["fv_direction"]))
                if a_f is not None and "fv_direction" in a_f.files:
                    g["cos_fv"] = float(w @ _unit(a_f["fv_direction"]))
                lw = (ext_wd.get("layers") or {}).get(str(l)) or {}
                ll = (ext_l.get("layers") or {}).get(str(l)) or {}
                g["stability_wd"] = (ext_wd.get("stability") or {}).get(str(l))
                g["stability_d31"] = (ext_l.get("stability") or {}).get(str(l))
                g["final_loss_wd"] = (lw.get("final_loss") or [None])[0]
                g["final_loss_d31"] = (ll.get("final_loss") or [None])[0]
                g["initial_loss"] = (lw.get("initial_loss") or [None])[0]
                g["stationarity_wd"] = (lw.get("stationarity") or [None])[0]
                g["cos_with_init_wd"] = (lw.get("cos_with_init") or [None])[1:]
                geometry[str(l)] = g
        row["geometry"] = geometry
        out["tasks"][name] = row
    return out


def _f(x: Any, nd: int = 2) -> str:
    return "–" if x is None else (f"{x:+.{nd}f}" if isinstance(x, float) else str(x))


def markdown(res: dict[str, Any]) -> str:
    lines = ["| task | qualified wd / d31 / fv | effect wd / d31 / fv | wd layer, rho; norm/radius | cos(wd, d31) | cos(wd, fv) | "
             "cos(d31, fv) | cos(wd, PC1) | stability wd / d31 | neutral KL wd / d31 / fv |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, row in res["tasks"].items():
        wd, l, fv = row["wd"], row["learned"], row["fv"]
        layer = wd.get("layer")
        g = row["geometry"].get(str(layer)) if layer is not None else None
        g = g or {}
        lines.append(
            f"| {name} | {'/'.join('y' if r['qualified'] else 'n' for r in (wd, l, fv))} | "
            f"{' / '.join(_f(r.get('effect')) for r in (wd, l, fv))} | "
            f"{layer if layer is not None else '–'}, {_f(wd.get('rho'))}; {_f(g.get('fitted_norm_over_radius'))} | "
            f"{_f(g.get('cos_d31'))} | {_f(g.get('cos_fv'))} | {_f(g.get('cos_d31_fv'))} | {_f(g.get('cos_pca'))} | "
            f"{_f(g.get('stability_wd'))} / {_f(g.get('stability_d31'))} | "
            f"{' / '.join(_f(r.get('neutral_kl_mean'), 3) for r in (wd, l, fv))} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wd-run", required=True, type=Path)
    ap.add_argument("--learned-run", required=True, type=Path)
    ap.add_argument("--fv-run", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    res = compare(args.wd_run, args.learned_run, args.fv_run)
    if args.out:
        args.out.write_text(json.dumps(res, indent=1, default=float))
    print(markdown(res))


if __name__ == "__main__":
    main()
