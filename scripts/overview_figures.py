"""Overview figures of the two variable-map lines, two designs for each (descriptive; no new readouts).

Reads `docs/overview/data.json` (written by `scripts/overview_data.py`) and draws four figures into
`docs/overview/`, for Qwen3 4B, 8B and 14B Base, with depth as the fraction of the stack (read point / L: the 4B
and 8B have 36 blocks, the 14B 40, and their landmarks line up by fraction rather than by block):

- `arith_route.png`  (A1) addition/subtraction, D45-D54, as a route map: each model's operands travelling along
  depth from their digit tokens to the answer token, and which sub-results the answer token then holds as separate
  variables (DAS), with the line's key numbers beside each route;
- `arith_panels.png` (A2) the same line as evidence panels: the measured curves per model (map, probe, DAS cells)
  on a shared depth axis, then the cross-model readings (DAS in the Cause-Isolate plane, the count, the block split,
  the token sets);
- `maxmin_route.png`  (B1) max/min, D55-D60, as a route map: the operator (the control) and the two sums (the
  computation) travelling through the query's token groups until they meet at `=` and the answer position;
- `maxmin_panels.png` (B2) the same line as evidence panels: the token-set swaps of the operator and of the sums,
  the direction's switch curve, and the max DAS (D55 -> D56) in the Cause-Isolate plane.

Every mark is a recorded reading: sufficiency and presence use the entries' lower-bound rules, held variables are
the DAS summaries' own `held` sets, and the landmarks are the summaries' values (labelled `read point/L`).

    uv run python scripts/overview_figures.py docs/overview/data.json --out docs/overview
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
JOINT = "#b9b7ae"  # the answer token's joint code: no sum separable, Y not yet its own variable
VAR = {"S1": BLUE, "S2": ORANGE, "S3": AQUA, "Y": INK, "T": MUTED, "U": MUTED}
MODELS = ("qwen3_4b", "qwen3_8b", "qwen3_14b")
SIZE = {"qwen3_4b": "4B", "qwen3_8b": "8B", "qwen3_14b": "14B"}
DASH = (0, (4, 2.2))  # three brackets; the added direction
DEPTH = "relative depth (read point / L)"
# Y decodes from its held subspace: D48's amendment reads "Y held" as Y itself only where it does (two brackets reach
# 0.9 and above there), and D51 reads three brackets' 0.71-0.79 the same way; 0.7 separates both from the joint code's
# 0.24-0.60 (a descriptive cut for the figure, not a preregistered one)
Y_DECODES = 0.7


def style():
    import matplotlib

    matplotlib.use("Agg")
    import logging

    import matplotlib.pyplot as plt

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({
        "font.family": ["Inter", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlesize": 9.5, "axes.titleweight": "semibold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "legend.frameon": False, "legend.fontsize": 8, "lines.solid_capstyle": "round", "svg.fonttype": "none"})
    return plt


def bezier(x0, y0, x1, y1, n=24):
    """A horizontal S-curve from (x0, y0) to (x1, y1)."""
    t = np.linspace(0, 1, n)
    xm = (x0 + x1) / 2
    xs = (1 - t) ** 3 * x0 + 3 * (1 - t) ** 2 * t * xm + 3 * (1 - t) * t ** 2 * xm + t ** 3 * x1
    ys = (1 - t) ** 3 * y0 + 3 * (1 - t) ** 2 * t * y0 + 3 * (1 - t) * t ** 2 * y1 + t ** 3 * y1
    return xs, ys


def depth_axis(ax, step=0.1, label=True):
    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.round(np.arange(0, 1.0001, step), 2))
    if label:
        ax.set_xlabel(DEPTH)


# ----------------------------------------------------------------------------------------------------------------
# A1: the arithmetic line as a route map


def split_label(r: dict) -> str:
    """D47 in a few words: how the brackets arrive at block G (heads) and where the sum becomes readable."""
    sp = r.get("split")
    if not sp:
        return ""
    heads = sp["heads"].get(str(r["G"]), {}).get("reading", "")
    heads = {"lanes": "a head per bracket", "mixed": "heads: mixed", "no carrier": "no single head"}.get(heads, heads)
    gains = {int(b): v for b, v in sp["probe"]["S1|Y"].items() if v["reading"] != "no gain"}
    if not gains:
        form = "sums readable gradually"
    else:
        b = max(gains, key=lambda k: gains[k]["gain"])
        form = {"mlp": "MLP adds", "attention": "attention-led", "mixed": "attention + MLP"}[gains[b]["reading"]]
        form += f" (block {b})"
    return f"{heads};\n{form}"


def fig_arith_route(data: dict, path: Path) -> None:
    plt = style()

    rows = {(r["task"], r["model"]): r for r in data["arith"]["rows"]}
    das = data["arith"]["das"]
    posset = data["arith"]["posset"]
    tracks = [{"task": t, "model": m, "row": rows[(t, m)], "das": das[t][m]} for t in ("two_lane", "three_lane")
              for m in MODELS]
    fig = plt.figure(figsize=(16.5, 8.6))
    ax = fig.add_axes([0.075, 0.19, 0.60, 0.665])
    tab = fig.add_axes([0.685, 0.19, 0.305, 0.665], sharey=ax)
    tab.axis("off")
    y = 0.0
    for i, tr in enumerate(tracks):
        if i == len(MODELS):
            y -= 0.75  # the gap between the two tasks
        tr["y"] = y
        y -= 1.0
    ys = [tr["y"] for tr in tracks]
    ax.set_xlim(-0.015, 1.045)
    ax.set_ylim(ys[-1] - 0.75, ys[0] + 1.05)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks(np.round(np.arange(0, 1.0001, 0.1), 1))
    ax.set_xlabel(DEPTH + " — the 4B and 8B have L = 36 blocks, the 14B 40")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for label, y0 in (("Two brackets   (a+b) − (c+d) =", ys[0]),
                      ("Three brackets   (a+b) − (c+d) − (e+f) =", ys[len(MODELS)])):
        ax.text(-0.015, y0 + 0.62, label, fontsize=10.5, fontweight="semibold", color=INK, va="bottom")

    for tr in tracks:
        r, d, y0, task = tr["row"], tr["das"], tr["y"], tr["task"]
        L, G, last = r["L"], r["G"], r["last_digit"]
        f = 1.0 / L
        ax.text(-0.022, y0, SIZE[tr["model"]], ha="right", va="center", fontsize=10, fontweight="semibold",
                color=INK, clip_on=False)
        brackets = ("S1", "S2", "S3") if task == "three_lane" else ("S1", "S2")
        off = dict(zip(brackets, (0.24, 0.0, -0.24) if len(brackets) == 3 else (0.17, -0.17)))
        ax.add_patch(plt.Rectangle((0, y0 - 0.36), 1.0, 0.72, fc="#f3f2ee", ec="none", zorder=0))
        ax.text(1.008, y0, f"L {L}", va="center", fontsize=7.5, color=MUTED)
        sites = sorted(int(s) for s in d["sites"])
        post = [s for s in sites if s >= G]
        held = {s: {v for v, c in d["sites"][str(s)].items() if c["held"]} for s in post}

        # the operands at their digit tokens: thin strands, opacity the share of bases in which a same-sum swap at the
        # bracket's digit tokens changes the answer (D45 map)
        mp = r["map"]
        for b in brackets:
            rate = np.asarray(mp["digits_same"][b], dtype=float)
            for m in range(min(last, L)):
                a = float(np.clip((rate[m] + rate[m + 1]) / 2, 0.12, 1))
                ax.plot([m * f, (m + 1) * f], [y0 + off[b]] * 2, color=VAR[b], lw=1.6, alpha=a, zorder=2)
            # to the answer token: a strand held at G stays its own line, the others merge into the joint code
            target = off[b] if (post and b in held[post[0]]) else 0.0
            xs, yy = bezier(last * f, y0 + off[b], G * f, y0 + target)
            ax.plot(xs, yy, color=VAR[b], lw=1.6, alpha=0.9, zorder=2)
        if tr["model"] == MODELS[0]:
            for bb, lab in zip(brackets, ("a + b", "c + d", "e + f")):
                ax.text(0.01, y0 + off[bb] + 0.035, lab, fontsize=6.5, color=INK2, va="bottom")
        # the gather
        ax.plot([G * f] * 2, [y0 - 0.42, y0 + 0.42], color=INK, lw=1.2, zorder=4)
        ax.text(G * f, y0 + 0.47, f"G {G}/{L}", ha="center", va="bottom", fontsize=7.5, color=INK,
                fontweight="semibold")

        # the answer token from G on: the joint code (gray) or Y as its own variable (ink), per DAS site, with held
        # sums as their own thick strands
        def center_state(s):
            c = d["sites"][str(s)].get("Y")
            return "Y" if c and c["held"] and (c["dec"] or 0) >= Y_DECODES else "joint"

        for s, s_next in zip(post, post[1:] + [None]):
            col = INK if center_state(s) == "Y" else JOINT
            if s_next is None:
                ax.plot([s * f, 1.0], [y0, y0], color=col, lw=2.2, zorder=3, ls=(0, (0.1, 2.2)),
                        dash_capstyle="round")
            else:
                ax.plot([s * f, s_next * f], [y0, y0], color=col, lw=4.2, zorder=3, solid_capstyle="butt")
            for b in brackets:
                if b in held[s]:
                    nxt = s_next if s_next is not None else min(s + 1, L)
                    if s_next is not None and b in held[s_next]:
                        ax.plot([s * f, s_next * f], [y0 + off[b]] * 2, color=VAR[b], lw=3.2, zorder=3)
                    else:
                        xs, yy = bezier(s * f, y0 + off[b], nxt * f, y0)
                        ax.plot(xs, yy, color=VAR[b], lw=3.2, zorder=3)
        if post and post[0] > G:
            ax.plot([G * f, post[0] * f], [y0, y0], color=JOINT, lw=4.2, zorder=3)
        # stations: every DAS site at or after G; held sums as dots on their strands, Y as a dot on the center
        # (filled where Y decodes from its subspace, a ring where it does not: the joint code passes the Y test)
        for s in post:
            ax.plot([s * f] * 2, [y0 - 0.30, y0 + 0.30], color=INK2, lw=0.8, zorder=2.5)
            for b in brackets:
                if b in held[s]:
                    src = d["sites"][str(s)][b]["source"]
                    ax.plot(s * f, y0 + off[b], "o", ms=7.5, mfc=VAR[b] if "extension" not in src else SURFACE,
                            mec=VAR[b], mew=2, zorder=5)
            c = d["sites"][str(s)].get("Y")
            if c and c["held"]:
                dec = (c["dec"] or 0) >= Y_DECODES
                ax.plot(s * f, y0, "o", ms=6.5, mfc=INK if dec else SURFACE, mec=INK, mew=1.6, zorder=5)
        # the DAS control, before the gather: nothing held there on any model
        ax.plot(d["control"] * f, y0, marker="x", ms=5, mew=1.3, color=MUTED, zorder=4)
        # where the Y probe reaches 0.9 of its final accuracy (D46): the answer linearly formed
        ax.plot(r["F"] * f, y0 - 0.33, marker="^", ms=6, color=INK, mec=SURFACE, mew=0.8, zorder=5)

    # the table beside the routes
    T = tab.get_yaxis_transform()
    cols = [0.0, 0.115, 0.45, 0.70]
    heads = ["few-shot\naccuracy", "ceiling: the whole state\nswapped at G (D45, D52)",
             "sums held apart\nat G (DAS)", "how the sums form\nat G (D47)"]
    for x, h in zip(cols, heads):
        tab.text(x, ys[0] + 0.62, h, transform=T, fontsize=8, color=INK2, va="bottom", fontweight="semibold")
    for tr in tracks:
        r, d, y0, task = tr["row"], tr["das"], tr["y"], tr["task"]
        tab.text(cols[0], y0, f"{r['accuracy']:.2f}", transform=T, va="center", fontsize=9, color=INK)
        # the whole-state ceiling: two brackets from the D45 map; three brackets from D52's 256 bases, the answer
        # token alone and with `=`
        if task == "two_lane":
            bars = [("", r["ceiling_at_G"], INK2)]
        else:
            ps = posset[tr["model"]]
            k = ps["read_points"].index(ps["marks"]["G3"])
            bars = [("answer token", ps["sets"]["answer"]["mean"][k], INK2),
                    ("with `=`", ps["sets"]["answer_eq"]["mean"][k], MUTED)]
        for j, (lab, v, col) in enumerate(bars):
            yy = y0 + (0.14 - 0.28 * j if len(bars) == 2 else 0)
            tab.add_patch(plt.Rectangle((cols[1], yy - 0.09), 0.12 * v, 0.18, transform=T, color=col, lw=0))
            tab.add_patch(plt.Rectangle((cols[1], yy - 0.09), 0.12, 0.18, transform=T, fc="none", ec=AXIS, lw=0.6))
            tab.text(cols[1] + 0.13, yy, f"{v:.2f}", transform=T, va="center", fontsize=8, color=INK)
            if lab:
                tab.text(cols[1] + 0.185, yy, lab, transform=T, va="center", fontsize=7, color=MUTED)
        G = r["G"]
        held = [v for v in ("S1", "S2", "S3") if d["sites"].get(str(G), {}).get(v, {}).get("held")]
        expl = any("extension" in d["sites"][str(G)][v]["source"] for v in held)
        tab.text(cols[2], y0 + 0.06, f"{len(held)}", transform=T, va="center", fontsize=15, fontweight="semibold",
                 color=INK)
        x = cols[2] + 0.045
        for b in held:
            tab.plot(x, y0 + 0.06, "o", transform=T, ms=7, mfc=SURFACE if expl else VAR[b], mec=VAR[b], mew=2)
            tab.text(x + 0.014, y0 + 0.06, b, transform=T, va="center", fontsize=7.5, color=INK2)
            x += 0.06
        note = {("two_lane", "qwen3_4b"): "exploratory (D48 extension)",
                ("two_lane", "qwen3_14b"): "S1, S2 Cause ≤ 0.29",
                ("two_lane", "qwen3_8b"): "for one block"}.get((task, tr["model"]), "0 with `=` too (D54)")
        tab.text(cols[2], y0 - 0.24, note, transform=T, va="center", fontsize=7, color=MUTED)
        tab.text(cols[3], y0, split_label(r), transform=T, va="center", fontsize=7.5, color=INK2, linespacing=1.3)

    fig.text(0.075, 0.95, "Addition and subtraction (D45–D54): the operands' route to the answer token, and what it "
             "holds there", fontsize=14, fontweight="semibold", color=INK)
    fig.text(0.075, 0.92, "Qwen3 4B, 8B and 14B Base, seed 20260916, 8-shot. Each track is one model's forward pass "
             "along relative depth; strands are the brackets, the center line is the answer token after the operands "
             "gather there.", fontsize=9.5, color=INK2)
    # the key: two rows of four
    k = fig.add_axes([0.075, 0.04, 0.915, 0.08])
    k.axis("off")
    k.set_xlim(0, 100)
    k.set_ylim(0, 3)

    def item(x, y, kind, text):
        if kind == "strand":
            k.plot([x, x + 3.5], [y + 0.18, y + 0.18], color=BLUE, lw=1.6)
            k.plot([x, x + 3.5], [y - 0.18, y - 0.18], color=ORANGE, lw=1.6)
        elif kind == "thick":
            k.plot([x, x + 3.5], [y, y], color=BLUE, lw=3.2)
            k.plot(x + 1.75, y, "o", ms=7.5, mfc=BLUE, mec=BLUE)
        elif kind == "ring":
            k.plot([x, x + 3.5], [y, y], color=BLUE, lw=3.2)
            k.plot(x + 1.75, y, "o", ms=7.5, mfc=SURFACE, mec=BLUE, mew=2)
        elif kind == "joint":
            k.plot([x, x + 3.5], [y, y], color=JOINT, lw=4.2, solid_capstyle="butt")
        elif kind == "Y":
            k.plot([x, x + 3.5], [y, y], color=INK, lw=4.2, solid_capstyle="butt")
            k.plot(x + 1.75, y, "o", ms=6.5, mfc=INK, mec=INK)
        elif kind == "Yring":
            k.plot([x, x + 3.5], [y, y], color=JOINT, lw=4.2, solid_capstyle="butt")
            k.plot(x + 1.75, y, "o", ms=6.5, mfc=SURFACE, mec=INK, mew=1.6)
        elif kind == "dotted":
            k.plot([x, x + 3.5], [y, y], color=INK, lw=2.2, ls=(0, (0.1, 2.2)), dash_capstyle="round")
        elif kind == "marks":
            k.plot([x + 0.6, x + 0.6], [y - 0.35, y + 0.35], color=INK, lw=1.2)
            k.plot(x + 1.9, y, marker="x", ms=5, mew=1.3, color=MUTED)
            k.plot(x + 3.1, y, marker="^", ms=6, color=INK)
        k.text(x + 4.4, y, text, fontsize=7.5, color=INK2, va="center", linespacing=1.25)

    item(0, 2.3, "strand", "a bracket's digits at their own tokens (D45);\nopacity: same-sum swaps there change the answer")
    item(0, 0.7, "thick", "a sum held as its own variable (DAS: Cause ≥ 0.5,\nIsolate ≥ 0.75, lower bounds); dot: held site")
    item(26, 2.3, "ring", "the same, from D48's exploratory extension\n(k ≤ 64, 20 epochs)")
    item(26, 0.7, "joint", "the answer token's joint code: no sum separable,\nY not yet its own variable; ticks: DAS sites")
    item(52, 2.3, "Y", "Y held as its own variable, and Y decodes from\nthe held subspace (≥ 0.7; chance 0.12–0.25)")
    item(52, 0.7, "Yring", "passes the Y test but Y does not decode:\nstill the joint code (D48's amendment)")
    item(78, 2.3, "dotted", "after the last DAS site (not tested)")
    item(78, 0.7, "marks", "G gather (D45/D51) · × DAS control\n▲ Y probe at 0.9 of its final accuracy")
    fig.text(0.075, 0.01, "Not drawn: the one-lane twin (a+b) − c gathers at the same read point as two brackets on "
             "each model. D53's joint DAS over 12 tokens overfit and moved the answer as much at its control as at the "
             "gather (Makelov-type illusion), so it is not read. One seed.", fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------
# A2: the arithmetic line as evidence panels


def fig_arith_panels(data: dict, path: Path) -> None:
    plt = style()
    from matplotlib.lines import Line2D

    rows = {(r["task"], r["model"]): r for r in data["arith"]["rows"]}
    das = data["arith"]["das"]
    fig = plt.figure(figsize=(15, 15.5))
    outer = fig.add_gridspec(2, 1, left=0.095, right=0.985, top=0.83, bottom=0.085, hspace=0.2,
                             height_ratios=[3.1, 1.25])
    gs = outer[0].subgridspec(3, 3, hspace=0.13, wspace=0.12)
    row_titles = [
        ("Map (D45, D51)", "whole-state swap: share of bases\nwhose answer moves"),
        ("Probe (D46, D51)", "linear readout of the answer\ntoken, cross-validated accuracy"),
        ("DAS (D48–D51)", "test Cause of each fitted\nsubspace at the answer token"),
    ]
    for i, m in enumerate(MODELS):
        two, three = rows[("two_lane", m)], rows[("three_lane", m)]
        L, G = two["L"], two["G"]
        f = 1.0 / L
        axes = [fig.add_subplot(gs[j, i]) for j in range(3)]
        for j, ax in enumerate(axes):
            depth_axis(ax, step=0.25, label=j == 2)
            ax.set_ylim(-0.02, 1.04)
            ax.set_yticks([0, 0.5, 1])
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            if i:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(row_titles[j][1], fontsize=8)
                ax.text(-0.27, 0.5, row_titles[j][0], transform=ax.transAxes, rotation=90, va="center",
                        ha="center", fontsize=10.5, fontweight="semibold", color=INK)
            ax.axvline(G * f, color=INK, lw=0.8, alpha=0.5, zorder=0)
        axes[0].set_title(f"Qwen3-{SIZE[m]} Base   (G = {G}/{L} = {G * f:.2f})", fontsize=10.5)
        # 1. the map: the answer token (the combined swap gives the source's answer) and the digit tokens (a same-sum
        # swap there changes the answer)
        for r, ls in ((two, "-"), (three, DASH)):
            mp = r["map"]
            x = np.asarray(mp["read_points"]) / r["L"]
            comb = mp["answer"]["all" if r["task"] == "three_lane" else "both"]
            axes[0].fill_between(x, comb["lo"], comb["hi"], color=INK, alpha=0.07, lw=0)
            axes[0].plot(x, comb["mean"], color=INK, lw=1.8, ls=ls)
            dig = np.max([mp["digits_same"][b] for b in mp["digits_same"]], axis=0)
            axes[0].plot(x, dig, color=MUTED, lw=1.8, ls=ls)
        # 2. the probe: S1 given Y (the sum beyond the answer) and Y, with the Y probe's 0.9 point (F)
        for r, ls in ((two, "-"), (three, DASH)):
            pr = r["probe"]
            x = np.asarray(pr["read_points"]) / r["L"]
            axes[1].plot(x, pr["probes"]["S1|Y"]["accuracy"], color=BLUE, lw=1.8, ls=ls)
            axes[1].plot(x, pr["probes"]["Y"]["accuracy"], color=INK, lw=1.8, ls=ls)
            axes[1].plot(r["F"] / r["L"], 0.0, marker="^", ms=6, color=INK, clip_on=False, zorder=5,
                         mfc=INK if ls == "-" else SURFACE)
        # 3. DAS: Cause at k* per site, filled where the site's reading holds the variable; the map's whole-state
        # ceiling at the answer token behind it
        for r, task, dx, mk, ls in ((two, "two_lane", -0.004, "o", "-"), (three, "three_lane", 0.004, "^", DASH)):
            comb = r["map"]["answer"]["all" if task == "three_lane" else "both"]
            axes[2].plot(np.asarray(r["map"]["read_points"]) * f, comb["mean"], color=JOINT, lw=1.4, ls=ls, zorder=1)
            d = das[task][m]
            sites = sorted(int(s) for s in d["sites"])
            for v in ("S1", "S2", "S3", "Y"):
                pts = [(s, d["sites"][str(s)][v]) for s in sites if v in d["sites"][str(s)]]
                if not pts:
                    continue
                xs = [s * f + dx for s, _ in pts]
                axes[2].plot(xs, [c["cause"] for _, c in pts], color=VAR[v], lw=1.0, ls=ls, alpha=0.8, zorder=2)
                for (s, c), x in zip(pts, xs):
                    axes[2].plot(x, c["cause"], marker=mk, ms=6.5, mec=VAR[v], mew=1.4, zorder=3,
                                 mfc=VAR[v] if c["held"] else SURFACE)
            axes[2].plot(d["control"] * f + dx, 0.0, marker="x", ms=5, mew=1.2, color=MUTED, clip_on=False)
        for ax in axes[:2]:
            ax.set_xticklabels([])
        # direct labels on the 8B column, where the lines are apart
        if m == "qwen3_8b":
            arrow = dict(arrowstyle="-", color=AXIS, lw=0.6)
            axes[0].annotate("answer token", (27 * f, 0.93), xytext=(28.5 * f, 0.62), fontsize=7.5, color=INK2,
                             arrowprops=arrow)
            axes[0].annotate("digit tokens", (14 * f, 0.86), xytext=(4 * f, 0.62), fontsize=7.5, color=INK2,
                             arrowprops=arrow)
            axes[1].annotate("S1 given Y", (24.5 * f, 0.9), xytext=(9 * f, 0.83), fontsize=7.5, color=INK2,
                             arrowprops=arrow)
            axes[1].annotate("Y", (26.5 * f, 0.97), xytext=(30 * f, 0.55), fontsize=7.5, color=INK2,
                             arrowprops=arrow)
            axes[2].text(31.5 * f, 0.80, "ceiling: the\nwhole state", fontsize=7, color=MUTED, ha="center")

    # the summary row
    b = outer[1].subgridspec(1, 4, wspace=0.42, width_ratios=[0.75, 1.25, 1.05, 0.9])
    # (a) the count
    ax = fig.add_subplot(b[0])
    count = {}
    for task in ("two_lane", "three_lane"):
        for m in MODELS:
            site = das[task][m]["sites"].get(str(rows[(task, m)]["G"]), {})
            held = [v for v in ("S1", "S2", "S3") if site.get(v, {}).get("held")]
            count[(task, m)] = (len(held), any("extension" in site[v]["source"] for v in held))
    for m in MODELS:
        G = rows[("three_lane", m)]["G"]
        sh = das["three_lane_eq"][m]["shared"]["sites"].get(str(G), {})
        jo = das["three_lane_eq"][m]["joint"]["sites"].get(str(G), {})
        count[("three_lane_eq", m)] = (max(sum(1 for v in ("S1", "S2", "S3") if c.get(v, {}).get("held"))
                                           for c in (sh, jo)), False)
    for (task, mk, ls, dx, lab) in (("two_lane", "o", "-", -0.12, "two brackets, answer token (D48–D50)"),
                                    ("three_lane", "^", DASH, 0.0, "three brackets, answer token (D51)"),
                                    ("three_lane_eq", "s", ":", 0.12, "three brackets, answer token + `=` (D54)")):
        xs = list(range(len(MODELS)))
        ys = [count[(task, m)][0] for m in MODELS]
        ax.plot([x + dx for x in xs], ys, color=INK2, lw=1.0, ls=ls, zorder=1)
        for x, y in zip(xs, ys):
            expl = count[(task, MODELS[x])][1]
            ax.plot(x + dx, y, marker=mk, ms=8, mfc=SURFACE if expl else INK, mec=INK, mew=1.4, zorder=3, ls="")
    ax.annotate("exploratory\n(D48 extension)", (-0.12, 1), xytext=(-0.42, 0.38), fontsize=7, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.6))
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([SIZE[m] for m in MODELS])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylim(-0.3, 3.3)
    ax.set_xlim(-0.5, len(MODELS) - 0.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_title("Sums held as separate\nvariables at the gather", fontsize=9.5)
    ax.legend(handles=[Line2D([], [], marker=mk, ls=ls, color=INK2, mfc=INK, mec=INK, ms=7, label=lab)
                       for mk, ls, lab in (("o", "-", "two brackets, answer token (D48–D50)"),
                                           ("^", DASH, "three brackets, answer token (D51)"),
                                           ("s", ":", "three brackets, answer token + `=` (D54)"))],
              loc="upper left", bbox_to_anchor=(-0.02, -0.12), fontsize=7.5, handlelength=2.2)
    # (b) every DAS cell from the gather on, in the Cause-Isolate plane
    ax = fig.add_subplot(b[1])
    ax.add_patch(plt.Rectangle((0.75, 0.5), 0.27, 0.52, color="#eef4fc", lw=0, zorder=0))
    ax.text(0.765, 0.99, "held", ha="left", va="top", fontsize=8, color=BLUE, fontweight="semibold")
    for task, mk in (("two_lane", "o"), ("three_lane", "^")):
        for m in MODELS:
            r, d = rows[(task, m)], das[task][m]
            for s, site in d["sites"].items():
                for v, c in site.items():
                    if v not in VAR:
                        continue
                    if int(s) < r["G"]:
                        ax.plot(c["iso_lo"], c["lo"], marker="x", ms=4, mew=0.9, color=AXIS, zorder=1)
                        continue
                    sum_var = v.startswith("S")
                    ax.plot(c["iso_lo"], c["lo"], marker=mk, ms=6 if sum_var else 4.5, ls="", zorder=3 if sum_var else 2,
                            mfc=VAR[v] if c["held"] else SURFACE, mec=VAR[v], mew=1.2, alpha=1 if sum_var else 0.75)
                    if c["held"] and sum_var:
                        tag = f"{SIZE[m]} {v} @ {s}/{r['L']}" + (" (expl.)" if "extension" in c["source"] else "")
                        dxy, ha = {("qwen3_4b", "S1"): ((-7, -3), "right"), ("qwen3_8b", "S1"): ((5, 5), "left"),
                                   ("qwen3_8b", "S2"): ((-7, -3), "right")}.get((m, v), ((5, 5), "left"))
                        ax.annotate(tag, (c["iso_lo"], c["lo"]), xytext=dxy, ha=ha, textcoords="offset points",
                                    fontsize=7, color=INK2)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.axvline(0.75, color=AXIS, lw=0.7)
    ax.axhline(0.5, color=AXIS, lw=0.7)
    ax.set_xlabel("Isolate, lower bound (the answer stays when only\nanother variable changes)")
    ax.set_ylabel("Cause, lower bound")
    ax.set_title("Every DAS cell from the gather on (× before it)", fontsize=9.5)
    handles = [Line2D([], [], marker="o", ls="", mfc=VAR[v], mec=VAR[v], ms=6, label=v) for v in ("S1", "S2", "S3")]
    handles += [Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, ms=4.5, label="Y"),
                Line2D([], [], marker="o", ls="", mfc=MUTED, mec=MUTED, ms=4.5, label="T, U (merges)"),
                Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK2, ms=6, label="not held"),
                Line2D([], [], marker="^", ls="", mfc=SURFACE, mec=INK2, ms=6, label="three brackets")]
    ax.legend(handles=handles, loc="upper left", fontsize=7, ncol=2, handletextpad=0.2, columnspacing=0.8)
    # (c) the block split
    ax = fig.add_subplot(b[2])
    items = []
    for task in ("two_lane", "three_lane"):
        for m in MODELS:
            sp = rows[(task, m)]["split"]["probe"]
            gains = {int(k): v for k, v in sp["S1|Y"].items() if v["reading"] != "no gain"}
            ygains = {int(k): v for k, v in sp["Y"].items() if v["reading"] != "no gain"}
            items.append((task, m, gains, ygains))
    for k, (task, m, gains, ygains) in enumerate(items):
        y = len(items) - 1 - k + (0 if task == "three_lane" else 0.6)
        ax.text(-0.06, y, f"{SIZE[m]}  {'3 br.' if task == 'three_lane' else '2 br.'}", ha="right", va="center",
                fontsize=8, color=INK2, transform=ax.get_yaxis_transform())
        if gains:
            bb = max(gains, key=lambda q: gains[q]["gain"])
            g = gains[bb]
            lo, hi = g["ci"]
            ax.plot([lo, hi], [y + 0.12, y + 0.12], color=BLUE, lw=2, solid_capstyle="round")
            ax.plot(g["share"], y + 0.12, "o", ms=7, color=BLUE, mec=SURFACE, mew=1)
            ax.text(min(hi, 1.0) + 0.03, y + 0.12, f"block {bb}", fontsize=7, color=MUTED, va="center")
        else:
            ax.text(0.02, y + 0.12, "S1 given Y rises over several blocks (no step)", fontsize=7, color=MUTED,
                    va="center")
        if ygains:
            yb = max(ygains, key=lambda q: ygains[q]["gain"])
            ax.plot(ygains[yb]["share"], y - 0.18, "D", ms=5.5, color=INK, mec=SURFACE, mew=0.8)
    ax.axvspan(-0.25, 0.5, color="#f3f2ee", lw=0, zorder=0)
    ax.text(0.12, len(items) + 0.4, "MLP", fontsize=8, color=INK2, ha="center", fontweight="semibold")
    ax.text(0.8, len(items) + 0.4, "attention", fontsize=8, color=INK2, ha="center", fontweight="semibold")
    ax.set_xlim(-0.25, 1.15)
    ax.set_ylim(-0.7, len(items) + 0.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("attention's share of the block's gain (95% interval)")
    ax.set_title("Where the sums form (D47): the gather block", fontsize=9.5)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=6, label="S1 given Y: largest-gain block"),
                       Line2D([], [], marker="D", ls="", color=INK, ms=5, label="Y: largest-gain block")],
              loc="upper left", bbox_to_anchor=(-0.02, -0.17), fontsize=7.5, ncol=2)
    # (d) D52: the token sets at the three-bracket gather
    ax = fig.add_subplot(b[3])
    sets = (("answer", "answer token", INK2), ("answer_eq", "answer token + `=`", MUTED),
            ("non_digit", "all 13 non-digit tokens", AXIS))
    for k, m in enumerate(MODELS):
        ps = data["arith"]["posset"][m]
        j = ps["read_points"].index(ps["marks"]["G3"])
        y = len(MODELS) - 1 - k
        vals = [ps["sets"][key]["mean"][j] for key, _, _ in sets]
        ax.plot([min(vals), max(vals)], [y, y], color=AXIS, lw=1.2, zorder=1)
        for (key, lab, col), v in zip(sets, vals):
            ax.plot(v, y, "o", ms=8, color=col, mec=SURFACE, mew=1, zorder=3, label=lab if k == 0 else None)
        ax.text(0.405, y, SIZE[m], ha="right", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0.42, 1.0)
    ax.set_ylim(-0.6, 3.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("every sum changed: share of bases\ngiving the source's answer, at G3")
    ax.set_title("Where the rest of the\nanswer lives (D52)", fontsize=9.5)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.03, 1.02), fontsize=7.5)

    fig.text(0.07, 0.957, "Addition and subtraction (D45–D54): the evidence, model by model", fontsize=14,
             fontweight="semibold", color=INK)
    fig.text(0.07, 0.935, "Qwen3 4B, 8B and 14B Base, seed 20260916, on relative depth (read point / L). Solid: two "
             "brackets (a+b) − (c+d) =; dashed: three brackets (a+b) − (c+d) − (e+f) =.\nVertical line: the gather G "
             "(D45). Shaded: 95% bootstrap interval.", fontsize=9.5, color=INK2, va="top", linespacing=1.4)
    leg = [Line2D([], [], color=INK, lw=1.8, label="map: answer token, every sum swapped → source's answer"),
           Line2D([], [], color=MUTED, lw=1.8, label="map: digit tokens, same-sum swap → answer changes"),
           Line2D([], [], color=BLUE, lw=1.8, label="probe: S1 given Y"),
           Line2D([], [], color=INK, lw=1.8, marker="^", ms=6, label="probe: Y (▲ 0.9 of final, F)"),
           Line2D([], [], color=JOINT, lw=1.4, label="DAS row: the whole-state ceiling")]
    leg += [Line2D([], [], marker="o", ls="", mfc=VAR[v], mec=VAR[v], ms=6.5, label=f"DAS {v}") for v in
            ("S1", "S2", "S3", "Y")]
    leg += [Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK2, mew=1.4, ms=6.5, label="not held (filled: held)"),
            Line2D([], [], marker="x", ls="", color=MUTED, ms=5, label="DAS control site")]
    fig.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.07, 0.905), ncol=4, fontsize=7.8,
               handlelength=2.2, columnspacing=1.4)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------
# B1: the max/min line as a route map

LANES = (("O", "operator token  max/min"), ("A", "answer position"), ("E", "` =`"), ("P", "( , ␣ )"),
         ("FS", "operands  a+b, c+d"))
LANE_Y = {g: len(LANES) - 1 - i for i, (g, _) in enumerate(LANES)}
PAIRS = (("O", "A", "OA"), ("A", "E", "EA"), ("E", "P", "PE"))


def lane_panel(ax, rp, rates, color, L, o_early=None):
    """Token-group lanes along relative depth: a ribbon per lane, half-width the share of bases whose answer a swap of
    that group alone moves; between adjacent lanes, a band whose opacity is what the pair moves beyond either alone."""
    x = np.asarray(rp, dtype=float) / L
    for g, _ in LANES:
        y = LANE_Y[g]
        ax.axhline(y, color=GRID, lw=0.8, zorder=0)
        if g not in rates:
            continue
        xg, v = x, np.asarray(rates[g]["mean"], dtype=float)
        if g == "O" and o_early is not None:
            ex, ev = o_early
            keep = ex < rp[0]
            xg, v = np.concatenate([ex[keep] / L, x]), np.concatenate([ev[keep], v])
        w = 0.40 * np.clip(v, 0, 1)
        ax.fill_between(xg, y - w, y + w, color=color, alpha=0.9, lw=0, zorder=3)
    for g1, g2, pair in PAIRS:
        if pair not in rates:
            continue
        ex = np.asarray(rates[pair]["mean"]) - np.maximum(rates[g1]["mean"], rates[g2]["mean"])
        y1, y2 = LANE_Y[g1], LANE_Y[g2]
        for k in range(len(x) - 1):
            a = float(np.clip((ex[k] + ex[k + 1]) / 2, 0, 1))
            if a > 0.04:
                ax.fill_between([x[k], x[k + 1]], min(y1, y2), max(y1, y2), color=color, alpha=0.8 * a, lw=0,
                                zorder=1)
    # read points the token-set swaps did not cover (before them, only the operator token lane is measured, by D58)
    for x0, x1, only_o in ((0, x[0], o_early is not None), (x[-1], 1.0, False)):
        if x1 > x0:
            hi = LANE_Y["O"] - 0.5 if only_o else len(LANES) - 0.5
            ax.fill_between([x0, x1], -0.5, hi, color="#f3f2ee", lw=0, zorder=0.5)
    depth_axis(ax, label=False)
    ax.set_ylim(-0.55, len(LANES) - 0.45)
    ax.set_yticks([LANE_Y[g] for g, _ in LANES])
    ax.set_yticklabels([lab for _, lab in LANES], fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def station(ax, x, y, label, color=INK, above=True, ha="center"):
    ax.plot(x, y, "o", ms=7, mfc=SURFACE, mec=color, mew=1.8, zorder=6)
    ax.annotate(label, (x, y), xytext=(0, 8 if above else -9), textcoords="offset points", ha=ha,
                va="bottom" if above else "top", fontsize=7.5, color=INK, fontweight="semibold", zorder=7)


def fig_maxmin_route(data: dict, path: Path) -> None:
    plt = style()
    from matplotlib.lines import Line2D

    mm = data["maxmin"]
    fig = plt.figure(figsize=(16.5, 14.2))
    outer = fig.add_gridspec(3, 1, left=0.19, right=0.745, top=0.885, bottom=0.085, hspace=0.3)
    for i, m in enumerate(MODELS):
        o = mm["operator"][m]
        L, lm, ts = o["L"], o["landmarks"], o["two_site"]
        f = 1.0 / L
        g = outer[i].subgridspec(3, 1, hspace=0.12, height_ratios=[1, 1, 0.62])
        ax_o, ax_s, ax_d = (fig.add_subplot(g[k]) for k in range(3))
        rp = o["opset"]["read_points"]
        c = o["curve"]
        early = (np.asarray(c["read_points"], dtype=float), np.asarray(c["arms"]["swap"]["mean"], dtype=float))
        lane_panel(ax_o, rp, o["opset"]["natural"]["rates"], ORANGE, L, early)
        lane_panel(ax_s, rp, o["opset"]["sums"]["rates"], BLUE, L)
        for ax in (ax_o, ax_s):
            ax.set_xticklabels([])
        ax_o.set_title(f"Qwen3-{SIZE[m]} Base   (L = {L})", fontsize=11)
        ax_o.text(-0.17, 0.5, "the operator\n(max ↔ min)", transform=ax_o.transAxes, ha="right", va="center",
                  fontsize=9.5, fontweight="semibold", color=INK)
        ax_s.text(-0.17, 0.5, "the two sums\n(both changed)", transform=ax_s.transAxes, ha="right", va="center",
                  fontsize=9.5, fontweight="semibold", color=INK)
        # the operator's landmarks: it leaves its token (O_out, D57), `=` + the answer position take it (T2, D60), the
        # answer position alone (R_O, D57); where the direction was added (D58)
        yO, yA, yE, yP = LANE_Y["O"], LANE_Y["A"], LANE_Y["E"], LANE_Y["P"]
        station(ax_o, lm["O_out"] * f, yO, f"leaves its token  {lm['O_out']}/{L}", ha="left")
        station(ax_o, ts["T2"] * f, (yA + yE) / 2, f"`=` + answer  {ts['T2']}/{L}", above=False)
        if lm["R_O"] < L:
            station(ax_o, lm["R_O"] * f, yA, f"answer alone  {lm['R_O']}/{L}")
        else:
            ax_o.text(0.99, yA + 0.42, "answer position alone:\nnot before the output", ha="right", va="bottom",
                      fontsize=7, color=MUTED)
        ax_o.plot(c["inject"] * f, yO + 0.55, marker="v", ms=7, color=INK, zorder=6, clip_on=False)
        ax_o.text(c["inject"] * f + 0.012, yO + 0.55, "v added", fontsize=7, color=INK2, va="center")
        # the sums' landmarks: `=` + answer (G2, D60), the answer position alone (G, D57)
        station(ax_s, ts["G2"] * f, (yA + yE) / 2, f"`=` + answer  {ts['G2']}/{L}", above=False)
        if lm["G"] < L:
            station(ax_s, lm["G"] * f, yA, f"answer alone  {lm['G']}/{L}")
        else:
            ax_s.text(0.99, yA + 0.42, "answer position alone:\nnot before the output", ha="right", va="bottom",
                      fontsize=7, color=MUTED)
        lead = ts["G2"] - ts["T2"]
        ax_s.plot([ts["T2"] * f] * 2, [yP - 0.25, (yA + yE) / 2], color=ORANGE, lw=1.2, ls=(0, (1, 1.5)), zorder=5)
        ax_s.plot(ts["T2"] * f, (yA + yE) / 2, "o", ms=5, mfc=SURFACE, mec=ORANGE, mew=1.5, zorder=6)
        ax_s.annotate("", xy=(ts["T2"] * f, yP), xytext=(ts["G2"] * f, yP),
                      arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9, shrinkA=0, shrinkB=0))
        ax_s.text(ts["T2"] * f - 0.01, yP, f"operator there first, by {lead} blocks ({lead * f:.2f})", ha="right",
                  va="center", fontsize=7.5, color=INK2)
        # the answer token on max alone (D55): Cause of each fitted subspace per DAS site, filled = held
        key = "qwen3_14b/shared" if m == "qwen3_14b" else f"{m}/answer"
        d55 = mm["d55"][key]
        mx = mm["max"][m]
        G55 = mx.get("G_star", mx["G"])
        ax_d.axvline(G55 * f, color=INK, lw=0.8, alpha=0.5)
        for v, mk, col in (("S1", "o", BLUE), ("S2", "s", BLUE), ("C", "D", MAGENTA), ("Y", "o", INK)):
            pts = sorted((int(s_), site["variables"][v]["fit"]["cause"], v in site["held"])
                         for s_, site in d55["sites"].items() if v in site["variables"])
            ax_d.plot([x * f for x, _, _ in pts], [y for _, y, _ in pts], color=col, lw=0.8, alpha=0.6)
            for x, y, h in pts:
                ax_d.plot(x * f, y, marker=mk, ms=5.5, mfc=col if h else SURFACE, mec=col, mew=1.3, zorder=3)
        depth_axis(ax_d, label=i == len(MODELS) - 1)
        ax_d.set_ylim(0, 1.05)
        ax_d.set_yticks([0, 1])
        ax_d.grid(axis="y", color=GRID, lw=0.6)
        ax_d.text(-0.17, 0.5, "DAS at the answer\ntoken, max alone\n(D55): test Cause", transform=ax_d.transAxes,
                  ha="right", va="center", fontsize=8.5, color=INK2)
        top = m != "qwen3_14b"  # the 14B's control sits just before its G*, high in the strip
        ax_d.text(G55 * f - 0.01, 1.0 if top else 0.04, f"max gather {G55}/{L}" + ("" if top else
                                                                                    " (G*; DAS spans `=` too)"),
                  fontsize=7, color=INK2, va="top" if top else "bottom", ha="right")
        if i == 0:
            ax_d.legend(handles=[Line2D([], [], marker=mk, ls="", mfc=col, mec=col, ms=5.5, label=lab) for lab, mk, col
                                 in (("S1", "o", BLUE), ("S2", "s", BLUE), ("C", "D", MAGENTA), ("Y", "o", INK))]
                        + [Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK2, ms=5.5, label="not held")],
                        loc="upper left", ncol=5, fontsize=7.5, handletextpad=0.2, columnspacing=0.9)
        # the readings beside each model
        d58 = {"qwen3_4b": "only early", "qwen3_8b": "one direction while the token holds it",
               "qwen3_14b": "only early"}[m]
        nat, add = o["opset"]["natural"]["rates"], o["opset"]["added"]["rates"]
        fs_max = max(max(nat["FS"]["mean"]), max(add["FS"]["mean"]))
        cs = d55["sites"][str(d55["control_site"])]["variables"]["C"]["fit"]["cause"]
        lines = [
            ("max/min few-shot accuracy", f"{o['map']['accuracy']:.2f}"),
            ("operand tokens carry the operator", f"never (≤ {fs_max:.2f})"),
            ("the direction at its token (D58)", d58),
            (f"added at read point {c['inject']}: the route", "the natural one (D58, D59)"),
            ("sums held apart at the gather (D55)", "0" + ("; D56 too" if m != "qwen3_14b" else "")),
            ("comparison C at the DAS control", f"Cause {cs:.2f}, not held"),
            ("Y held from", "the gather" + (" + 1" if m == "qwen3_8b" else "") + " (D55)"),
        ]
        yy = ax_o.get_position().y1 - 0.004
        for lab, val in lines:
            fig.text(0.765, yy, lab, fontsize=7.8, color=MUTED, va="top")
            fig.text(0.765, yy - 0.0115, val, fontsize=8.6, color=INK, va="top", fontweight="semibold")
            yy -= 0.0305

    fig.text(0.11, 0.958, "Max and min (D55–D60): the operator and the sums travel apart and meet at `=` and the "
             "answer position", fontsize=14, fontweight="semibold", color=INK)
    fig.text(0.11, 0.94, "Qwen3 4B, 8B and 14B Base, seed 20260916, on relative depth (read point / L); max(a+b, c+d) "
             "= and min(a+b, c+d) = mixed in each prompt.\nLanes are the query's token groups; swaps of a set of groups "
             "between a base and its twin (D59, D60).", fontsize=9.5, color=INK2, va="top", linespacing=1.4)
    k = fig.add_axes([0.19, 0.012, 0.78, 0.05])
    k.axis("off")
    k.set_xlim(0, 100)
    k.set_ylim(0, 2)
    k.fill_between([0, 1.6], 1.2, 1.8, color=ORANGE, lw=0)
    k.fill_between([1.8, 3.4], 1.2, 1.8, color=BLUE, lw=0)
    k.text(4.2, 1.5, "ribbon width: share of bases in which swapping that token group alone switches max ↔ min "
           "(operator) or gives the source's answer (sums)", fontsize=7.8, color=INK2, va="center")
    k.fill_between([0, 1.6], 0.2, 0.8, color=ORANGE, alpha=0.35, lw=0)
    k.fill_between([1.8, 3.4], 0.2, 0.8, color=BLUE, alpha=0.35, lw=0)
    k.text(4.2, 0.5, "band between two lanes: what swapping the pair moves beyond either alone (the pair holds it "
           "jointly)        ○ landmark (first or last read point by the entries' lower-bound rules)",
           fontsize=7.8, color=INK2, va="center")
    k.fill_between([78, 80], 1.2, 1.8, color="#f3f2ee", lw=0)
    k.text(80.8, 1.5, "not covered by the set swaps", fontsize=7.8, color=INK2, va="center")
    fig.text(0.19, 0.003, "Before read point 8 the operator lane is D58's twin swap at the operator token; the set swaps "
             "cover read points 8 to L − 4.", fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------
# B2: the max/min line as evidence panels

SETS = (("O", "operator token alone", ORANGE), ("A", "answer position alone", INK), ("E", "` =` alone", MUTED),
        ("OA", "operator + answer (the two ends)", AQUA), ("EA", "` =` + answer", BLUE),
        ("FS", "operands a+b, c+d", YELLOW))
DAS_SHAPES = {"S1": ("o", BLUE), "S2": ("s", BLUE), "C": ("D", MAGENTA), "Y": ("o", INK)}


def fig_maxmin_panels(data: dict, path: Path) -> None:
    plt = style()
    from matplotlib.lines import Line2D

    mm = data["maxmin"]
    fig = plt.figure(figsize=(17, 16.5))
    gs = fig.add_gridspec(4, 4, left=0.075, right=0.985, top=0.885, bottom=0.05, hspace=0.36, wspace=0.2,
                          width_ratios=[1, 1, 1.08, 1.0])
    row_titles = [("The operator (D57, D59, D60)", "swap of a token set:\nshare switching max ↔ min"),
                  ("The two sums (D60)", "swap of a token set (both sums\nchanged): source's answer"),
                  ("The direction (D58)", "at the operator token:\nshare switching max → min"),
                  ("DAS on max alone (D55 → D56)", "Cause, lower bound")]
    for i, m in enumerate(MODELS):
        o = mm["operator"][m]
        L, lm, ts, c = o["L"], o["landmarks"], o["two_site"], o["curve"]
        f = 1.0 / L
        x = np.asarray(o["opset"]["read_points"]) * f
        axes = [fig.add_subplot(gs[j, i]) for j in range(4)]
        for j, ax in enumerate(axes[:3]):
            depth_axis(ax, step=0.2, label=j == 2)
            ax.set_ylim(-0.02, 1.04)
            ax.set_yticks([0, 0.5, 1])
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.axhline(0.5, color=AXIS, lw=0.6, zorder=0)
            if i:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(row_titles[j][1], fontsize=8)
        for j in range(4):
            if i == 0:
                axes[j].text(-0.33, 0.5, row_titles[j][0], transform=axes[j].transAxes, rotation=90, va="center",
                             ha="center", fontsize=10.5, fontweight="semibold", color=INK)
        axes[0].set_title(f"Qwen3-{SIZE[m]} Base   (L = {L})", fontsize=10.5)
        # 1. the operator: natural arm (solid) and the added direction's arm (dashed), with D58's twin swap at the
        # operator token before the set swaps start
        for arm, ls, lw, al in (("natural", "-", 1.8, 1.0), ("added", DASH, 1.2, 0.85)):
            rates = o["opset"][arm]["rates"]
            for key, lab, col in SETS:
                if arm == "added" and key not in ("O", "OA", "EA"):
                    continue
                if arm == "natural" and key in ("OA", "EA"):
                    axes[0].fill_between(x, rates[key]["lo"], rates[key]["hi"], color=col, alpha=0.12, lw=0)
                axes[0].plot(x, rates[key]["mean"], color=col, lw=lw, ls=ls, alpha=al)
        cr = np.asarray(c["read_points"])
        e = cr <= o["opset"]["read_points"][0]
        axes[0].plot(cr[e] * f, np.asarray(c["arms"]["swap"]["mean"])[e], color=ORANGE, lw=1.8)
        tag = dict(fontsize=7.5, color=INK2, bbox=dict(fc=SURFACE, ec="none", pad=0.6, alpha=0.9), zorder=6)
        axes[0].axvline(ts["T2"] * f, color=BLUE, lw=0.9, ls=(0, (1, 1.5)))
        axes[0].text(ts["T2"] * f + 0.008, 0.17, f"T2 {ts['T2']}/{L}", **tag)
        axes[0].axvline(lm["O_out"] * f, color=ORANGE, lw=0.9, ls=(0, (1, 1.5)))
        axes[0].text(lm["O_out"] * f - 0.008, 0.08, f"O_out {lm['O_out']}/{L}", ha="right", **tag)
        # 2. the sums
        rates = o["opset"]["sums"]["rates"]
        for key, lab, col in SETS:
            if key in ("O", "OA"):
                continue
            if key in ("EA", "FS"):
                axes[1].fill_between(x, rates[key]["lo"], rates[key]["hi"], color=col, alpha=0.12, lw=0)
            axes[1].plot(x, rates[key]["mean"], color=col, lw=1.8)
        axes[1].axvline(ts["G2"] * f, color=BLUE, lw=0.9, ls=(0, (1, 1.5)))
        axes[1].text(ts["G2"] * f + 0.008, 0.45, f"G2 {ts['G2']}/{L}", **tag)
        axes[1].axvline(ts["T2"] * f, color=ORANGE, lw=0.9, ls=(0, (1, 1.5)))
        axes[1].text(ts["T2"] * f - 0.008, 0.45, f"T2 {ts['T2']}/{L}", ha="right", **tag)
        # 3. the direction: added (dashed) against the twin's whole state at the operator token, and random nulls
        for arm, col, ls in (("swap", ORANGE, "-"), ("steer", ORANGE, DASH), ("null0", MUTED, "-"),
                             ("null1", MUTED, "-")):
            a = c["arms"][arm]
            if arm in ("swap", "steer"):
                axes[2].fill_between(cr * f, a["lo"], a["hi"], color=col, alpha=0.1, lw=0)
            axes[2].plot(cr * f, a["mean"], color=col, ls=ls, lw=1.8 if arm != "null1" else 1.0)
        axes[2].plot(c["inject"] * f, 1.04, marker="v", ms=7, color=INK, clip_on=False)
        axes[2].text(0.02, 0.12, f"▼ m_inj {c['inject']}/{L}: v added here\n(the steered map, the added arm)",
                     va="center", **tag)
        for w, dy in ((c["W_swap"], 0.80), (c["W_steer"], 0.70)):
            axes[2].plot([w * f] * 2, [0, dy], color=INK2, lw=0.8, ls=(0, (1, 1.5)))
        axes[2].text(max(c["W_swap"], c["W_steer"]) * f + 0.014, 0.74,
                     f"W_steer {c['W_steer']}/{L}, W_swap {c['W_swap']}/{L}", fontsize=7.5, color=INK2)
        for ax in axes[:2]:
            ax.set_xticklabels([])
        # 4. DAS on max, D55 -> D56, in the Cause-Isolate plane
        das_plane(axes[3], mm, m)
        axes[3].set_title(f"Qwen3-{SIZE[m]}", fontsize=9.5)
    # the summary column
    # (a) the take-up timeline, on relative depth
    ax = fig.add_subplot(gs[0, 3])
    for k, m in enumerate(MODELS):
        o = mm["operator"][m]
        lm, ts, L = o["landmarks"], o["two_site"], o["L"]
        f = 1.0 / L
        y = 2.2 - k
        ax.plot([0, 1], [y, y], color=GRID, lw=6, solid_capstyle="round", zorder=0)
        ax.plot([lm["O_out"] * f, ts["T2"] * f], [y, y], color=ORANGE, lw=1.4, zorder=1)
        ax.plot([ts["T2"] * f, ts["G2"] * f], [y, y], color=INK2, lw=1.4, zorder=1)
        ax.plot(lm["O_out"] * f, y, "o", ms=7, mfc=SURFACE, mec=ORANGE, mew=1.8, zorder=3)
        ax.plot(ts["T2"] * f, y, "o", ms=7, color=ORANGE, mec=SURFACE, mew=1, zorder=3)
        ax.plot(ts["G2"] * f, y, "s", ms=7, color=BLUE, mec=SURFACE, mew=1, zorder=3)
        ax.text(-0.03, y, SIZE[m], ha="right", va="center", fontsize=8.5, color=INK2)
        ax.text(1.02, y, f"L {L}", va="center", fontsize=7, color=MUTED)
        ax.text(ts["T2"] * f, y - 0.2, f"{ts['T2'] * f:.2f}", ha="center", va="top", fontsize=7, color=INK2)
        ax.text(ts["G2"] * f, y + 0.18, f"{ts['G2'] * f:.2f}", ha="center", fontsize=7, color=INK2)
    ax.set_xlim(-0.15, 1.12)
    ax.set_ylim(-1.75, 4.1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks(np.round(np.arange(0, 1.0001, 0.2), 1))
    ax.set_xlabel(DEPTH)
    tu = mm["take_up"]
    ax.set_title("Arrivals at ` =` + answer (D60)", fontsize=9.5)
    ax.text(-0.14, -0.75, f"14B vs the 4B/8B mean, in blocks: ΔG = {tu['dG']:g}, ΔT = {tu['dT']:g}\n→ "
            f"\"{tu['reading']}\", at the rule's limit; by fraction the\noperator arrives at a nearly fixed depth "
            "(0.56, 0.56, 0.58)", fontsize=7.3, color=INK2, va="top")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=ORANGE, mew=1.8, ms=7,
                              label="operator leaves its token (O_out, D57)"),
                       Line2D([], [], marker="o", ls="", color=ORANGE, ms=7, label="operator at ` =` + answer (T2)"),
                       Line2D([], [], marker="s", ls="", color=BLUE, ms=7, label="both sums at ` =` + answer (G2)")],
              loc="upper left", bbox_to_anchor=(0, 1.02), fontsize=7.3, handletextpad=0.3)
    # (b) the comparison readable before the sums (D55 probe, one block before the max gather)
    ax = fig.add_subplot(gs[1, 3])
    for k, m in enumerate(MODELS):
        mx = mm["max"][m]
        G = mx.get("G_star", mx["G"])
        pr = mx["probe"]["probes"]
        y = len(MODELS) - 1 - k
        for key, mk, col in (("C|Y", "D", MAGENTA), ("S1|Y", "o", BLUE), ("S2|Y", "s", BLUE)):
            v, nul = pr[key]["accuracy"][G - 1], pr[key]["null_mean"][G - 1]
            ax.plot([nul, v], [y, y], color=col, lw=1, alpha=0.5)
            ax.plot(nul, y, "|", ms=8, color=col, mew=1.2)
            ax.plot(v, y, mk, ms=7, color=col, mec=SURFACE, mew=0.8, zorder=3)
        ax.text(0.17, y, f"{SIZE[m]}  (G−1 = {G - 1}/{mx['L']})", ha="right", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0.18, 1.0)
    ax.set_ylim(-0.6, len(MODELS) - 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("probe accuracy at the answer token (| = its null)")
    ax.set_title("C is readable before the sums (D55 probe)", fontsize=9.5)
    ax.legend(handles=[Line2D([], [], marker="D", ls="", color=MAGENTA, ms=6, label="C given Y"),
                       Line2D([], [], marker="o", ls="", color=BLUE, ms=6, label="S1 given Y"),
                       Line2D([], [], marker="s", ls="", color=BLUE, ms=6, label="S2 given Y")],
              loc="lower left", fontsize=7.5, ncol=3, bbox_to_anchor=(-0.02, -0.03), handletextpad=0.2)
    # (c) the direction is one direction across prompts at its token
    ax = fig.add_subplot(gs[2, 3])
    for m, ls in zip(MODELS, ("-", DASH, ":")):
        c = mm["operator"][m]["curve"]
        cos = c["pair_cosine"]
        ax.plot(np.arange(len(cos)) / mm["operator"][m]["L"], cos, color=ORANGE, ls=ls, lw=1.6, label=SIZE[m])
    depth_axis(ax, step=0.2)
    ax.set_ylim(0.5, 1.02)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_title("Per-prompt min − max differences vs\ntheir mean, at the operator token (D58)", fontsize=9.5)
    ax.set_ylabel("mean cosine", fontsize=8)
    ax.legend(loc="lower left", fontsize=7.5, ncol=3)
    # (d) how to read the DAS planes
    ax = fig.add_subplot(gs[3, 3])
    das_key(ax)

    fig.text(0.075, 0.957, "Max and min (D55–D60): the evidence, model by model", fontsize=14, fontweight="semibold",
             color=INK)
    fig.text(0.075, 0.935, "Qwen3 4B, 8B and 14B Base, seed 20260916, on relative depth (read point / L). Rows 1–3: "
             "max(a+b, c+d) = and min(…) mixed in each prompt (D57–D60); row 4: max alone (D55, D56). Shaded: 95% "
             "bootstrap interval; dotted verticals: landmarks.", fontsize=9.5, color=INK2)
    leg = [Line2D([], [], color=col, lw=1.8, label=lab) for _, lab, col in SETS]
    leg += [Line2D([], [], color=INK2, lw=1.2, ls=DASH, label="dashed: the added direction (D58's v at m_inj)"),
            Line2D([], [], color=MUTED, lw=1.8, label="row 3 gray: random directions of v's norm")]
    fig.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.075, 0.925), ncol=4, fontsize=8,
               handlelength=2.4, columnspacing=1.6)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def das_plane(ax, mm: dict, m: str) -> None:
    """D55's cells (faint) and, where it ran, D56's (solid) at the max gather and, for C, at the DAS control, in the
    Cause-Isolate plane (lower bounds), joined by an arrow; the held region shaded."""
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((0.75, 0.5), 0.27, 0.52, color="#eef4fc", lw=0, zorder=0))
    ax.axvline(0.75, color=AXIS, lw=0.7)
    ax.axhline(0.5, color=AXIS, lw=0.7)
    d56 = mm["d56"].get(m)
    key = "qwen3_14b/shared" if m == "qwen3_14b" else f"{m}/answer"
    d55 = mm["d55"][key]
    mx = mm["max"][m]
    G, L = mx.get("G_star", mx["G"]), mx["L"]
    ctl = d55["control_site"]
    for site, at in ((G, "G"), (ctl, "ctl")):
        for v, (mk, col) in DAS_SHAPES.items():
            if at == "ctl" and v != "C":
                continue
            fit = d55["sites"][str(site)]["variables"][v]["fit"]
            x0, y0 = fit["isolate_lo"], fit["cause_ci"][0]
            ax.plot(x0, y0, mk, ms=6.5, color=col, alpha=0.4 if d56 else 1.0, mec=SURFACE, mew=0.6, zorder=3)
            xl, yl = x0, y0
            if d56:
                cell = d56["cells"][f"{site}/{v}"]["d56"]
                xl, yl = cell["isolate_lo"], cell["cause_lo"]
                ax.annotate("", xy=(xl, yl), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, shrinkA=4, shrinkB=4,
                                            mutation_scale=8))
                ax.plot(xl, yl, mk, ms=7, color=col, mec=SURFACE, mew=0.8, zorder=4)
            ax.annotate(f"{v}" + (f" @ ctl {ctl}/{L}" if at == "ctl" else ""), (xl, yl),
                        xytext=(5, -10) if v == "S2" else (5, 3), textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Isolate, lower bound")
    ax.text(0.765, 0.99, "held", ha="left", va="top", fontsize=8, color=BLUE, fontweight="semibold")
    ax.text(0.02, 0.02, f"at the gather {G}/{L}" + ("; faint D55 → solid D56" if d56 else "; D55 only (D56 not run)"),
            fontsize=7, color=MUTED)


def das_key(ax) -> None:
    """How to read the DAS planes of B2's last row."""
    from matplotlib.patches import Rectangle

    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reading the DAS planes", fontsize=9.5)
    ax.add_patch(Rectangle((0.02, 0.80), 0.10, 0.09, color="#eef4fc", lw=0))
    ax.text(0.16, 0.845, "held: Cause ≥ 0.5 and Isolate ≥ 0.75 (lower bounds),\nand Cause above the control (D54's rule)",
            fontsize=7.8, color=INK2, va="center")
    ax.plot(0.03, 0.66, "o", ms=6.5, color=BLUE, alpha=0.4, mec=SURFACE)
    ax.annotate("", xy=(0.11, 0.66), xytext=(0.03, 0.66),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.9, shrinkA=4, shrinkB=4, mutation_scale=8))
    ax.plot(0.11, 0.66, "o", ms=7, color=BLUE, mec=SURFACE)
    ax.text(0.16, 0.66, "D55 (trained on Cause) → D56 (trained on Cause\nand Isolate), the same bases and sources",
            fontsize=7.8, color=INK2, va="center")
    y = 0.47
    for v, (mk, col) in DAS_SHAPES.items():
        ax.plot(0.07, y, mk, ms=7, color=col, mec=SURFACE)
        ax.text(0.16, y, {"S1": "S1 = a + b", "S2": "S2 = c + d", "C": "C: the comparison S1 > S2",
                          "Y": "Y: the answer"}[v], fontsize=7.8, color=INK2, va="center")
        y -= 0.09
    ax.text(0.02, 0.06, "@ ctl: the DAS control site, before the gather; the\nother cells are at the max gather",
            fontsize=7.8, color=INK2, va="center")


FIGURES = {"arith_route.png": fig_arith_route, "arith_panels.png": fig_arith_panels,
           "maxmin_route.png": fig_maxmin_route, "maxmin_panels.png": fig_maxmin_panels}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", help="figure file names to draw (default: all)")
    args = ap.parse_args(argv)
    data = json.load(open(args.data))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        if args.only and name not in args.only:
            continue
        fn(data, args.out / name)
        print(f"wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
