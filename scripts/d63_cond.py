"""D63: the conditional task read by its preregistered rules, with one figure per model.

Per model, from the four parts' runs (cond_map_*, cond_sets_*, cond_pairs_*, cond_probe_*):
- Q1, where the condition is formed: held as a value (the flip present, the same-outcome source changing nothing)
  at h or a structural token right after it before it reaches the answer token alone (R_C) -> "formed at the
  condition"; first at = or the answer, or nowhere before R_C, while h holds it as a digit -> "deferred";
- Q2, where it meets the sums: K_CB against K_AB (together / compute, then select / select, then compute / never),
  the null (K_CN), and the localization;
- Q3, whether the unchosen sum is computed: the probe of S_un given Y carried at the answer position at a read point
  from G_ch - 2 to K_CB; K_CU - K_CB beside;
- beside them: the map's landmarks, the token sets' gap and its sufficient sets, the probes.

    uv run python scripts/d63_cond.py results/remote/run-<8B map+sets> results/remote/run-<8B pairs+probe> \
        results/remote/run-<14B> --out docs/cond/d63_cond.json --figures docs/cond
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

PARTS = ("map", "sets", "pairs", "probe", "steer")
PAIR_COL = {"CB": "#2a78d6", "CU": "#eb6834", "AB": "#1baf7a", "CN": "#8a5cd0", "SG": "#d9a521"}
PAIR_LABEL = {"CB": "condition x both sums", "CU": "condition x unchosen sum", "AB": "chosen branch: digit x digit",
              "CN": "same-outcome condition x chosen sum (null)", "SG": "sign x condition digits"}
SET_COL = {"Cd": "#d9a521", "A": "#0b0b0b", "E+A": "#eb6834", "Cd+A": "#2a78d6", "Bch+A": "#1baf7a", "P+E": "#c2417f"}


def find_runs(roots: list[Path]) -> dict[str, dict[str, Path]]:
    runs: dict[str, dict[str, Path]] = {}
    for root in roots:
        for run in sorted(root.glob("cond_*_qwen3_*")):
            m = re.match(r"cond_(map|sets|pairs|probe|steer)_qwen3_(\d+(?:\.\d+)?b)", run.name)
            if m and (run / "core" / f"{m.group(1)}.json").exists():
                runs.setdefault(m.group(2), {})[m.group(1)] = run
    return runs


def q1(read: dict, labels: list[str], read_points: list[int]) -> dict:
    """D63's Q1 from the map's landmarks."""
    R = read.get("R_C")
    limit = R if R is not None else read_points[-1] + 1
    h = labels.index("<h>")
    after = [labels[j] for j in range(h + 1, labels.index("<a>")) if labels[j].strip() in (",", "")]
    # the value test holds from h on: the same-outcome source keeps the g crossing by construction, so a site holding g
    # alone (or the sign) cannot change the answer under it whatever it holds
    tested = {t: m for t, m in read.get("value_C", {}).items() if t not in ("<g>", "<cmp>")}
    local = {t: m for t, m in tested.items() if (t == "<h>" or t in after) and m < limit}
    early = {t: m for t, m in tested.items() if m < limit}
    first = min(early.items(), key=lambda kv: kv[1]) if early else None
    h_digit = read.get("digit_C", {}).get("<h>")
    if local:
        reading = "formed at the condition"
    elif (first is None or first[0] in (" =", "␣")) and h_digit is not None:
        reading = "deferred"
    else:
        reading = "undecided"
    return {"reading": reading, "R_C": R, "value_before_R_C": early, "local": local, "h_as_digit": h_digit}


def q3(probe: dict | None, read_points: list[int], G_ch: int | None, K_CB: int | None) -> dict:
    if probe is None or G_ch is None or K_CB is None:
        return {"reading": "undecided (a part or a landmark is missing)"}
    p = probe["probes"].get("A/Sun|Y")
    carried = [m for m, c in zip(read_points, p["carried"]) if c]
    window = [m for m in carried if G_ch - 2 <= m <= K_CB]
    return {"reading": "the unchosen sum is computed" if window else "not decodable", "carried_at_A": carried,
            "window": [G_ch - 2, K_CB], "in_window": window}


def steer_reading(st: dict, R_C: int | None, C_out_h: int | None) -> dict:
    """D63's steering readings: at h, one direction if the added direction's switch reaches a lower bound of 0.5 with
    the null at most 0.05 everywhere, for as long as the token holds it if its span ends no more than one read point
    before the swap's; at the answer, the same from R_C on."""
    out = {"n": st["n"], "n_dir": st["n_dir"]}
    rps = st["read_points"]
    for site, a in st.get("arms", {}).items():
        lo = np.asarray(a["rates"]["steer"]["lo"])
        ok = lo >= 0.5 if site == "h" else (lo >= 0.5) & (np.asarray(rps) >= (R_C if R_C is not None else 10**6))
        clean = a["null_max"] <= 0.05
        first = int(rps[np.flatnonzero(ok)[0]]) if ok.any() else None
        last = int(rps[np.flatnonzero(ok)[-1]]) if ok.any() else None
        swap_last = a["span"]["swap"][1]
        if first is None or not clean:
            reading = "not one direction" if clean else "not interpretable (the null switches)"
        elif site == "h":
            reading = ("one direction for as long as the token holds it" if swap_last is None or last >= swap_last - 1
                       else "one direction only early")
        else:
            reading = "one direction at the answer"
        out[site] = {"reading": reading, "span_steer": [first, last], "span_swap": a["span"]["swap"],
                     "steer_max": round(a["steer_max"], 3), "null_max": round(a["null_max"], 3),
                     "switch": {int(m): round(float(x), 2) for m, x in zip(rps, a["rates"]["steer"]["mean"]) if m % 4 == 0},
                     "swap": {int(m): round(float(x), 2) for m, x in zip(rps, a["rates"]["swap"]["mean"]) if m % 4 == 0}}
    out["C_out_h_from_map"] = C_out_h
    return out


def read_model(parts: dict[str, Path]) -> dict:
    out: dict = {}
    data = {p: json.loads((run / "core" / f"{p}.json").read_text()) for p, run in parts.items()}
    out["runs"] = {p: str(run) for p, run in parts.items()}
    if "map" in data:
        m = data["map"]
        rps, labels = m["read_points"], m["labels"]
        ans = len(labels) - 1
        out["map"] = {"n_kept": m["n_kept"], "accuracy": round(m["accuracy"], 3),
                      **{k: v for k, v in m["readouts"].items() if k != "route"},
                      "flip_at_answer": {k: {int(rp): round(float(x), 2) for rp, x in zip(rps, np.asarray(r["flip"]["mean"])[:, ans]) if rp % 4 == 0}
                                         for k, r in m["rates"].items()},
                      "route": {int(k): v for k, v in m["readouts"].get("route", {}).items()}}
        out["Q1"] = q1(m["readouts"], labels, rps)
    if "sets" in data:
        s = data["sets"]
        out["sets"] = {k: {"n": a["n"], "first_sufficient": a["first_sufficient"], "last_sufficient": a["last_sufficient"],
                           "gap": a["gap"], "sufficient_in_gap": a["sufficient_in_gap"]} for k, a in s["arms"].items()}
    if "pairs" in data:
        pr = data["pairs"]
        rps = pr["read_points"]
        out["pairs"] = {"reading": pr["reading"], "per_pair": {}}
        for p, r in pr["pairs"].items():
            if "rates" not in r:
                out["pairs"]["per_pair"][p] = {"n": r["n"], "K": r.get("K")}
                continue
            add = r["rates"]["additive"]
            i0 = rps.index(r["K"]) if r.get("K") is not None else None
            out["pairs"]["per_pair"][p] = {
                "n": r["n"], "accuracy": {k: round(v, 3) for k, v in r["accuracy"].items()}, "K": r.get("K"),
                "additive_equals_joint_at_0": r["additive_equals_joint_at_0"],
                "joint_min": round(float(np.min(r["rates"]["joint"]["11"]["mean"])), 3),
                "additive_11": {int(m): round(float(x), 2) for m, x in zip(rps, add["11"]["mean"]) if m % 4 == 0},
                "at_K": None if i0 is None else {c: round(float(add[c]["mean"][i0]), 3) for c in pr["categories"]}}
    if "probe" in data:
        pb = data["probe"]
        rps = pb["read_points"]
        out["probe"] = {"n": pb["n"], "sites": pb["sites"],
                        "carried": {k: [m for m, c in zip(rps, v["carried"]) if c] for k, v in pb["probes"].items()},
                        "max_excess": {k: round(float(np.max(v["excess"])), 3) for k, v in pb["probes"].items()}}
    if "steer" in data:
        out["steer"] = steer_reading(data["steer"], out.get("map", {}).get("R_C"), out.get("map", {}).get("C_out", {}).get("<h>"))
    G_ch = out.get("map", {}).get("G_ch")
    K_CB = out.get("pairs", {}).get("reading", {}).get("K", {}).get("CB")
    out["Q3"] = q3(data.get("probe"), data["probe"]["read_points"] if "probe" in data else [], G_ch, K_CB)
    if "pairs" in data:
        out["Q2"] = {k: out["pairs"]["reading"][k] for k in ("order", "null_clear", "CU_minus_CB")} | {
            "localization": {p: [v["lowest"] for v in x.values()] for p, x in out["pairs"]["reading"]["localization"].items()}}
    return out


def figure(size: str, parts: dict[str, Path], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {p: json.loads((run / "core" / f"{p}.json").read_text()) for p, run in parts.items()}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.2))
    if "map" in data:
        m = data["map"]
        rps, labels = np.asarray(m["read_points"]), m["labels"]
        for ax, (k, stat, title) in zip(axes[0], (("cond", "flip", "condition flipped: answer moves to the other sum"),
                                                  ("cond_same", "change", "same outcome, h crossing flipped: answer changes"),
                                                  ("chosen", "flip", "chosen sum changed: answer moves to it"))):
            if k not in m["rates"]:
                ax.axis("off")
                continue
            im = ax.imshow(np.asarray(m["rates"][k][stat]["mean"]), aspect="auto", cmap="Blues", vmin=0, vmax=1,
                           interpolation="nearest", origin="lower")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([lab.strip("<>") for lab in labels], fontsize=7)
            step = max(1, len(rps) // 9)
            ax.set_yticks(range(0, len(rps), step))
            ax.set_yticklabels(rps[::step], fontsize=7)
            ax.set_ylabel("read point", fontsize=8)
            ax.set_title(f"map, {title} ({m['rates'][k]['n']} bases)", fontsize=8, loc="left")
            fig.colorbar(im, ax=ax, fraction=0.03)
    else:
        for ax in axes[0]:
            ax.axis("off")
    ax = axes[1, 0]
    if "pairs" in data:
        pr = data["pairs"]
        rps = np.asarray(pr["read_points"])
        for p, r in pr["pairs"].items():
            if "rates" not in r:
                continue
            f = r["rates"]["additive"]["11"]
            ax.fill_between(rps, f["lo"], f["hi"], color=PAIR_COL[p], alpha=0.15, lw=0)
            ax.plot(rps, f["mean"], color=PAIR_COL[p], lw=1.8, label=f"{PAIR_LABEL[p]} ({r['n']}), K = {r.get('K')}")
        ax.axhline(0.5, color="#52514e", lw=0.5, ls=":")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("read point", fontsize=8)
        ax.set_title(f"pairs: additive state's share of the joint change's answer; {pr['reading']['order']}", fontsize=8, loc="left")
        ax.legend(fontsize=6.5, frameon=False, loc="lower left")
        ax.tick_params(labelsize=7)
    else:
        ax.axis("off")
    ax = axes[1, 1]
    if "sets" in data and "cond" in data["sets"]["arms"]:
        s = data["sets"]
        rps = np.asarray(s["set_read_points"])
        a = s["arms"]["cond"]
        for name, col in SET_COL.items():
            f = a["rates"][name]
            ax.fill_between(rps, f["lo"], f["hi"], color=col, alpha=0.12, lw=0)
            ax.plot(rps, f["mean"], color=col, lw=1.6, label=name)
        ax.axhline(0.5, color="#52514e", lw=0.5, ls=":")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("read point", fontsize=8)
        ax.set_title(f"token sets: the condition swapped with a set ({a['n']} bases); gap {a['gap'][:1]}..{a['gap'][-1:]}",
                     fontsize=8, loc="left")
        ax.legend(fontsize=6.5, frameon=False, loc="upper left", ncol=2)
        ax.tick_params(labelsize=7)
    else:
        ax.axis("off")
    ax = axes[1, 2]
    if "probe" in data:
        pb = data["probe"]
        rps = np.asarray(pb["read_points"])
        for (k, col, ls) in (("A/C", "#2a78d6", "-"), ("A/C|Y", "#2a78d6", "--"), ("A/Sun|Y", "#eb6834", "-"),
                             ("Cd_last/C", "#d9a521", "-"), ("E/C", "#8a5cd0", "-"), ("E/Sun|Y", "#8a5cd0", "--")):
            v = pb["probes"][k]
            ax.plot(rps, v["z"], color=col, ls=ls, lw=1.5, label=f"{k} (threshold {v['threshold']:.1f})")
        ax.axhline(pb["probes"]["A/C"]["threshold"], color="#52514e", lw=0.5, ls=":")
        ax.set_xlabel("read point", fontsize=8)
        ax.set_title(f"probes: z against the permutation null ({pb['n']} examples)", fontsize=8, loc="left")
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
        ax.tick_params(labelsize=7)
    else:
        ax.axis("off")
    fig.suptitle(f"IF(g>h, a+b, c+d) in full (D63), Qwen3-{size.upper()}-Base", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", type=Path, nargs="+", help="directories holding cond_* runs")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--figures", type=Path, help="directory for qwen3_<size>_cond.png")
    args = ap.parse_args(argv)
    runs = find_runs(args.roots)
    if not runs:
        print("no cond runs found", file=sys.stderr)
        return 1
    summary = {}
    for size in sorted(runs, key=lambda x: float(x[:-1])):
        s = read_model(runs[size])
        summary[f"qwen3_{size}"] = s
        print(f"qwen3_{size}: parts {sorted(runs[size])}")
        for q in ("Q1", "Q2", "Q3", "steer"):
            if q in s:
                print(f"  {q}: {s[q]}")
        if "map" in s:
            print(f"  map: { {k: v for k, v in s['map'].items() if k not in ('route', 'flip_at_answer')} }")
            print(f"  map flips at the answer token: {s['map']['flip_at_answer']}")
        if "sets" in s:
            print(f"  sets: {s['sets']}")
        if "pairs" in s:
            for p, r in s["pairs"]["per_pair"].items():
                print(f"  pair {p}: {r}")
        if "probe" in s:
            print(f"  probe carried: {s['probe']['carried']}")
        if args.figures:
            figure(size, runs[size], args.figures / f"qwen3_{size}_cond.png")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
