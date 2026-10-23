"""The data behind the overview figures of the two variable-map lines (descriptive; no new readouts).

Collects, for Qwen3 4B, 8B and 14B Base (the models mapped on every task; the 1.7B misses the three-bracket and
max/min accuracy gates and the 0.6B the two-bracket one, so both are left out), the per-read-point curves the
overview figures draw and the landmarks and DAS cells the entries already record, into one file, so that
`scripts/overview_figures.py` runs from git alone:

- the arithmetic line (D45-D54): `(a+b) - (c+d)` (two-lane) and `(a+b) - (c+d) - (e+f)` (three-lane): the D45/D51
  map at the answer token and at the digit tokens, the D46/D51 probe and lens at the answer token, the D47 block
  split, the DAS cells of D48-D51 and D54 (D53 as its comparison), and D52's token sets;
- the max/min line (D55-D60): the D55 max map and probe, the DAS cells of D55 and D56, the D57 operator map, the D58
  switch curve, and the D59/D60 token-set swaps (natural, added and sums arms).

Curves are read from the run directories (`results/remote/run-<id>`, as downloaded by `scripts/runpod_log.py` or
`gh run download`); landmarks and DAS cells from the committed summaries under `docs/`. Rates are means over bases
with their 95% bootstrap intervals as stored by each stage, rounded to 3 decimals.

    uv run python scripts/overview_data.py results/remote --out docs/overview/data.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DOCS = Path(__file__).resolve().parent.parent / "docs"

MODELS = ("qwen3_4b", "qwen3_8b", "qwen3_14b")
# (task, model) -> (varmap run directory, varprobe run directory), relative to the downloads root
ARITH = {
    ("two_lane", "qwen3_4b"): ("run-38502381471/varmap_qwen3_4b_seed20260916",
                               "run-38587101804/varprobe_qwen3_4b_seed20260916"),
    ("two_lane", "qwen3_8b"): ("run-38738001665/varmap_qwen3_8b_seed20260916",
                               "run-38738001665/varprobe_qwen3_8b_seed20260916"),
    ("two_lane", "qwen3_14b"): ("run-38913588733/varmap_qwen3_14b_seed20260916",
                                "run-38913588733/varprobe_qwen3_14b_seed20260916"),
    ("three_lane", "qwen3_4b"): ("run-38934415363/varmap_three_qwen3_4b_seed20260916",
                                 "run-38934415363/varprobe_three_qwen3_4b_seed20260916"),
    ("three_lane", "qwen3_8b"): ("run-38934932951/varmap_three_qwen3_8b_seed20260916",
                                 "run-38934932951/varprobe_three_qwen3_8b_seed20260916"),
    ("three_lane", "qwen3_14b"): ("run-38934471957/varmap_three_qwen3_14b_seed20260916",
                                  "run-38934471957/varprobe_three_qwen3_14b_seed20260916"),
}
POSSET = {"qwen3_4b": "run-38980786015/posset_qwen3_4b_seed20260916",
          "qwen3_8b": "run-38980786015/posset_qwen3_8b_seed20260916",
          "qwen3_14b": "run-38980798799/posset_qwen3_14b_seed20260916"}
# the D55 max task: map and probe directories
MAX = {"qwen3_4b": ("run-39178177443/varmap_max_qwen3_4b_seed20260916",
                    "run-39178177443/varprobe_max_qwen3_4b_seed20260916"),
       "qwen3_8b": ("run-39178188546/varmap_max_qwen3_8b_seed20260916",
                    "run-39178188546/varprobe_max_qwen3_8b_seed20260916"),
       "qwen3_14b": ("run-39252850362/varmap_max_qwen3_14b_seed20260916",
                     "run-39252850362/varprobe_max_qwen3_14b_seed20260916")}
# the operator line: D57 map, D58 opsteer, D59/D60 opset (natural and added arms), D60 sums arm
OPERATOR = {"qwen3_4b": {"map": "run-39623460598/varmap_maxmin_qwen3_4b_seed20260916",
                         "opsteer": "run-39651059358/opsteer_qwen3_4b_seed20260916",
                         "opset": "run-39888455166/opset_qwen3_4b_seed20260916",
                         "sums": "run-40022353098/opset_sums_qwen3_4b_seed20260916"},
            "qwen3_8b": {"map": "run-39622530165/varmap_maxmin_qwen3_8b_seed20260916",
                         "opsteer": "run-39648774824/opsteer_qwen3_8b_seed20260916",
                         "opset": "run-39888184099/opset_qwen3_8b_seed20260916",
                         "sums": "run-40022347205/opset_sums_qwen3_8b_seed20260916"},
            "qwen3_14b": {"map": "run-40017790556/varmap_maxmin_qwen3_14b_seed20260916",
                          "opsteer": "run-40022343637/opsteer_qwen3_14b_seed20260916",
                          "opset": "run-40022343637/opset_qwen3_14b_seed20260916",
                          "sums": "run-40022343637/opset_qwen3_14b_seed20260916"}}
OPSET_SETS = ("O", "F", "S", "P", "E", "A", "OA", "EA", "FS", "PE", "FSPE", "OPEA", "OFSA", "PA", "SA", "FA")
BRACKETS = {"two_lane": ("S1", "S2"), "three_lane": ("S1", "S2", "S3"), "max": ("S1", "S2"), "maxmin": ("S1", "S2")}
DIGIT_LABELS = {"S1": ("a", "b"), "S2": ("c", "d"), "S3": ("e", "f")}


def r3(x):
    """Floats rounded to 3 decimals, recursively (NaN as None), so the file stays small and diffable."""
    if isinstance(x, dict):
        return {k: r3(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r3(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), 3)
    if isinstance(x, np.integer):
        return int(x)
    return x


def column(rate: dict, j: int) -> dict:
    """One token's column of a stored (read point x token) rate: mean, lo, hi over read points."""
    return {s: np.asarray(rate[s])[:, j] for s in ("mean", "lo", "hi")}


def map_curves(task_dir: Path, task: str) -> dict:
    """D45/D51/D55/D57 map at the answer token (each bracket swap's flip rate, the combined swap's source-answer rate,
    and the operator swap's where there is one), at `=`, and at the digit tokens (a same-sum swap there changes the
    answer: the operands held as digits)."""
    v = json.load(open(task_dir / "varmap.json"))
    labels, rates = v["labels"], v["rates"]
    ans, eq = len(labels) - 1, labels.index(" =")
    combined = "all" if "all" in rates else "both"
    out = {"read_points": v["read_points"], "labels": labels, "accuracy": v["accuracy"], "n_kept": v["n_kept"],
           "answer": {}, "eq": {}, "digits_same": {}}
    for k in (*BRACKETS[task], combined, "op"):
        if k in rates:
            out["answer"][k] = column(rates[k]["flip"], ans)
            out["eq"][k] = column(rates[k]["flip"], eq)
    if "op" in rates:
        out["op_token"] = column(rates["op"]["flip"], labels.index("op"))
    digits = [j for j, lab in enumerate(labels) if lab in "abcdef" and len(lab) == 1]
    for b in BRACKETS[task]:
        same = f"{b}_same"
        if same not in rates:
            continue
        cols = [j for j in digits if labels[j] in DIGIT_LABELS[b]]
        out["digits_same"][b] = np.asarray(rates[same]["change"]["mean"])[:, cols].max(axis=1)
    return out


def probe_curves(task_dir: Path) -> dict:
    """D46 conditional probe (accuracy, null mean, carried) and the logit lens's rate of Y, at the answer token."""
    p = json.load(open(task_dir / "varprobe.json"))
    out = {"read_points": p["read_points"], "n_kept": p["n_kept"], "probes": {}, "lens": {}}
    for t, pr in p["probes"].items():
        out["probes"][t] = {"accuracy": pr["accuracy"], "null_mean": pr["null_mean"], "carried": pr["carried"]}
    for c, lc in p["lens"]["candidates"].items():
        out["lens"][c] = {"rate": lc["rate"], "null_mean": lc["null_mean"], "carried": lc["carried"]}
    return out


def das_cells(site_entry: dict, source: str) -> dict:
    """One DAS site of a `das_summary.py` summary, per variable: test Cause at k* with its interval, the Isolate lower
    bound, k*, the shuffled-target null, Y-type decodability, and whether the site's reading held it."""
    held = set(site_entry.get("held", []))
    return {var: {"cause": c["cause"], "lo": c["cause_ci"][0], "hi": c["cause_ci"][1], "iso_lo": c["isolate_lo"],
                  "k": c["k_star"], "null": c.get("null"), "dec": c.get("decodability"),
                  "dec_chance": c.get("decodability_chance"), "held": var in held, "source": source}
            for var, c in site_entry["variables"].items()}


def arith_das() -> dict:
    """The answer-token DAS cells per task and model: two-lane D48 on the 4B (preregistered, and the exploratory
    extension where it ran the site, k up to 64 and 20 epochs, as D49 and D50 later fixed), D49, D50; three-lane D51;
    and D54's two-token fits (answer token and `=`, joint and shared) with its control rule."""
    das = DOCS / "das"
    out: dict = {"two_lane": {}, "three_lane": {}, "three_lane_eq": {}}
    pre, ext = json.load(open(das / "summary.json")), json.load(open(das / "summary_long.json"))
    m = "qwen3_4b"
    p, e = pre[f"das_{m}_seed20260916"], ext[f"das_long_{m}_seed20260916"]
    sites = {s: das_cells(c, "D48") for s, c in p["sites"].items()}
    for s, c in e["sites"].items():
        sites[s] = das_cells(c, "D48 extension")
    out["two_lane"][m] = {"control": p["control_site"], "sites": sites}
    for m, f, key, src in (("qwen3_8b", "summary_8b.json", "das_d49_qwen3_8b_seed20260916", "D49"),
                           ("qwen3_14b", "summary_14b.json", "das_d50_qwen3_14b_seed20260916", "D50")):
        d = json.load(open(das / f))[key]
        out["two_lane"][m] = {"control": d["control_site"],
                              "sites": {s: das_cells(c, src) for s, c in d["sites"].items()}}
    for m in MODELS:
        d = json.load(open(das / f"summary_three_{m.split('_')[1]}.json"))[f"das_d51_{m}_seed20260916"]
        out["three_lane"][m] = {"control": d["control_site"],
                                "sites": {s: das_cells(c, "D51") for s, c in d["sites"].items()}}
    d54 = json.load(open(das / "d54_compare.json"))
    for key, d in d54.items():
        m, variant = key.split("/")
        sites = {}
        for s, site in d["sites"].items():
            sites[s] = {var: {"cause": c["fit"]["cause"], "lo": c["fit"]["cause_ci"][0], "hi": c["fit"]["cause_ci"][1],
                              "iso_lo": c["fit"]["isolate_lo"], "k": c["fit"]["k_star"], "null": c["fit"]["null"],
                              "dec": c["fit"].get("decodability"), "held": var in site["held"],
                              "over_control": c.get("over_control"), "source": f"D54 {variant}"}
                        for var, c in site["variables"].items()}
        out["three_lane_eq"].setdefault(m, {})[variant] = {"control": d["control_site"], "sites": sites}
    return out


def arith(root: Path) -> dict:
    vm = {**json.load(open(DOCS / "varmap/summary.json")), **json.load(open(DOCS / "varmap/summary_three.json"))}
    vp = {**json.load(open(DOCS / "varprobe/summary.json"))}
    for m in ("4b", "8b", "14b"):
        vp[f"qwen3_{m}_three"] = json.load(open(DOCS / "varprobe/summary_three.json"))[f"qwen3_{m}"]
    vs = {}
    for f in ("summary.json", "summary_8b.json", "summary_14b.json", "summary_three_4b.json", "summary_three_8b.json",
              "summary_three_14b.json"):
        vs.update(json.load(open(DOCS / "varsplit" / f)))
    rows = []
    for (task, m), (map_dir, probe_dir) in ARITH.items():
        three = task == "three_lane"
        s = vm[f"varmap_{'three_' if three else ''}{m}_seed20260916"][task]
        row = {"task": task, "model": m, "L": s["n_read_points"] - 1, "accuracy": s["accuracy"],
               "G": s["as_run"]["gather_read_point"], "last_digit": s["as_run"]["last_digit_read_point"],
               "ceiling_at_G": s["as_run"]["both_swap_source_answer_at_gather"],
               "map": map_curves(root / map_dir / "core/tasks" / task, task)}
        ps = vp[f"{m}_three" if three else m][task]
        row["F"], row["lens_Y_from"] = ps["y_probe_formed_at"], ps["lens_reads_y_from"]
        row["probe"] = probe_curves(root / probe_dir / "core/tasks" / task)
        split = vs.get(f"varsplit_{'three_' if three else ''}{m}_seed20260916", {}).get(task)
        if split:
            row["split"] = {"probe": {t: split["probe"][t]["blocks"] for t in split["probe"]},
                            "heads": {b: {k: h[k] for k in ("reading", "all_heads")} for b, h in split["heads"].items()}}
        rows.append(row)
    posset = {}
    for m, d in POSSET.items():
        p = json.load(open(root / d / "core/tasks/three_lane/posset.json"))
        posset[m] = {"read_points": p["read_points"], "marks": p["marks"],
                     "sets": {k: column_set(p["rates"]["all"][k]["flip"]) for k in
                              ("answer", "answer_eq", "non_digit", "answer_digits", "query")},
                     "n": p["rates"]["all"]["n"]}
    d53 = json.load(open(DOCS / "das/d53_compare.json"))
    return {"rows": rows, "das": arith_das(), "posset": posset, "d53": d53}


def column_set(rate: dict) -> dict:
    return {s: np.asarray(rate[s]) for s in ("mean", "lo", "hi")}


def opset_arm(arm: dict) -> dict:
    return {"n": arm["n"], "rates": {k: column_set(arm["rates"][k]["flip"]) for k in OPSET_SETS if k in arm["rates"]}}


def maxmin(root: Path) -> dict:
    out: dict = {"max": {}, "operator": {}}
    sm = json.load(open(DOCS / "varmap/summary_max.json"))
    for m, (map_dir, probe_dir) in MAX.items():
        s = sm[f"varmap_max_{m}_seed20260916"]["max"]
        out["max"][m] = {"L": s["n_read_points"] - 1, "accuracy": s["accuracy"], "G": s["as_run"]["gather_read_point"],
                         "map": map_curves(root / map_dir / "core/tasks/max", "max"),
                         "probe": probe_curves(root / probe_dir / "core/tasks/max")}
    out["max"]["qwen3_14b"]["G_star"] = 30  # D55's 14B amendment: the both swap at the answer token first >= 0.5
    d57 = json.load(open(DOCS / "varmap/d57_operator.json"))
    d58 = json.load(open(DOCS / "varmap/d58_opsteer.json"))
    d60 = json.load(open(DOCS / "varmap/d60_14b.json"))
    for m, dirs in OPERATOR.items():
        mp = map_curves(root / dirs["map"] / "core/tasks/maxmin", "maxmin")
        st = json.load(open(root / dirs["opsteer"] / "core/opsteer.json"))
        op = json.load(open(root / dirs["opset"] / "core/opset.json"))
        sums = json.load(open(root / dirs["sums"] / "core/opset.json"))
        c = st["curve"]
        rec = {"L": len(mp["read_points"]) - 1, "map": mp,
               "landmarks": {k: d57[m]["map"]["as_run"][k] for k in ("G", "O_out", "R_O")} | {
                   "G_star": d57[m]["map"].get("G_star"), "reading": d57[m]["map"]["as_run"]["reading"]},
               "curve": {"read_points": c["read_points"], "n": c["n"], "W_steer": c["W_steer"], "W_swap": c["W_swap"],
                         "inject": st["inject_read_point"],
                         "arms": {a: column_set(c["arms"][a]["switch"]) for a in ("steer", "swap", "null0", "null1")},
                         "pair_cosine": st["direction"]["pair_cosine"]},
               "opset": {"read_points": op["read_points"], "natural": opset_arm(op["arms"]["natural"]),
                         "added": opset_arm(op["arms"]["added"]), "sums": opset_arm(sums["arms"]["sums"])},
               "two_site": d60["two_site"]["per_model"][m.split("_")[1]]}
        if m == "qwen3_14b":
            rec["landmarks"].update({k: d60["map"][k] for k in ("G", "R_O", "O_out", "G_star")})
        out["operator"][m] = rec
    out["take_up"] = d60["two_site"]["E+A"]
    out["d55"] = {k: v for k, v in json.load(open(DOCS / "das/d55_compare.json")).items() if k.split("/")[0] in MODELS}
    out["d56"] = {k: v for k, v in json.load(open(DOCS / "das/d56_vs_d55.json")).items() if k in MODELS}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="the downloads root holding run-<id> directories")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    data = {"arith": arith(args.root), "maxmin": maxmin(args.root)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r3(data), separators=(",", ":")) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
