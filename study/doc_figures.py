#!/usr/bin/env python3
"""Figures for the consolidated write-up in ``study/summary/``.

Every figure is drawn from data this repository can produce or from numbers recorded in ``NOTES`` /
``REPORT`` with the command that produced them; the provenance of each one is stated in the caption of
the Markdown document that embeds it.  Labels are English so that no CJK font is required.

Run::

    python3 -m study.doc_figures
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from study import cuspidal3r as c3  # noqa: E402

OUT = "study/summary/figures"


def fig_aspects() -> str:
    """The 3R's torus ``T^2 \\ Sigma`` labelled exactly: two aspects, one singular curve."""
    model = c3.build()
    grid = np.linspace(-np.pi, np.pi, 361)
    values = np.zeros((grid.size, grid.size))
    for i, q2 in enumerate(grid):
        for j, q3 in enumerate(grid):
            values[i, j] = c3.det_position(model, np.array([0.0, q2, q3]))
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=140)
    sign = np.sign(values)
    ax.pcolormesh(grid, grid, sign, cmap="coolwarm", vmin=-1, vmax=1, shading="auto")
    ax.contour(grid, grid, values, levels=[0.0], colors="black", linewidths=1.4)
    ax.set_xlabel("q2 (rad)")
    ax.set_ylabel("q3 (rad)")
    ax.set_title("3R: two aspects (red/blue), separated by the singular curve")
    ax.set_aspect("equal")
    fig.tight_layout()
    path = f"{OUT}/fig1_3r_aspects.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path}: aspect cells {int((sign > 0).sum())} / {int((sign < 0).sum())}, "
          f"singular cells {int((np.abs(values) < 1e-6).sum())}")
    return path


def fig_workspace() -> str:
    """The 3R workspace from the *validated* IK: solution count per pose, the three discriminant
    components, and the three pairwise tangencies (two of them cusp candidates)."""
    rho = np.linspace(0.0, 5.6, 380)
    z = np.linspace(-3.4, 4.6, 380)
    counts = np.zeros((z.size, rho.size))
    for i, zz in enumerate(z):
        for j, rr in enumerate(rho):
            r2, z2 = rr * rr, zz * zz
            a0 = 16 * r2 * r2 + 32 * r2 * z2 - 360 * r2 + 16 * z2 * z2 - 296 * z2 + 1769
            a1 = 16 * r2 * r2 + 32 * r2 * z2 - 168 * r2 + 16 * z2 * z2 - 104 * z2 + 185
            a2 = 16 * r2 * r2 + 32 * r2 * z2 - 264 * r2 + 16 * z2 * z2 - 200 * z2 + 401
            roots = np.roots([a0, 2.0 * a2, a1])
            positive = roots[(np.abs(roots.imag) < 1e-9) & (roots.real > 0)]
            counts[i, j] = 2 * positive.size
    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=140)
    mesh = ax.pcolormesh(rho, z, counts, cmap="viridis", shading="auto")
    fig.colorbar(mesh, ax=ax, label="real IK solutions at that tool point")
    # the three discriminant components, drawn in the (rho, z) plane
    rr, zz = np.meshgrid(rho, z, indexing="xy")
    r2, z2 = rr ** 2, zz ** 2
    comps = {
        "A0 = 0  (q3 = pi)": 16 * r2 * r2 + 32 * r2 * z2 - 360 * r2 + 16 * z2 * z2 - 296 * z2 + 1769,
        "A1 = 0  (q3 = 0)": 16 * r2 * r2 + 32 * r2 * z2 - 168 * r2 + 16 * z2 * z2 - 104 * z2 + 185,
        "C2 = 0  (shoulder)": 16 * r2 * r2 + 32 * r2 * z2 - 264 * r2 + 16 * z2 * z2 - 136 * z2 + 289,
    }
    for _label, field in comps.items():
        ax.contour(rr, zz, field, levels=[0.0], linewidths=1.1, linestyles="--", colors="w")
    for r, zc, marker, colour, name, offset in (
            (np.sqrt(0.5), 1.3229, "*", "red", "A1=C2 (0.14 deg)", (-6, -18)),
            (np.sqrt(12.5), 1.3229, "*", "orange", "A0=C2 (0.30 deg)", (8, 8)),
            (np.sqrt(6.5), 1.3229, "o", "white", "A0=A1 (interior)", (-58, 10))):
        ax.plot([r], [zc], marker=marker, color=colour, markersize=13, markeredgecolor="black")
        ax.plot([r], [-zc], marker=marker, color=colour, markersize=13, markeredgecolor="black")
        ax.annotate(name, (r, zc), textcoords="offset points", xytext=offset, color="black",
                    fontsize=7, bbox=dict(boxstyle="round", fc="white", alpha=0.85, lw=0))
    for label, _field in comps.items():
        ax.plot([], [], "w--", linewidth=1.1, label=label)
    ax.plot([], [], "*", color="red", label="cusp candidates (tangency measured)")
    ax.set_xlabel("rho (m)   [tool radius]")
    ax.set_ylabel("z (m)")
    ax.set_title("3R workspace from the validated IK quadratic")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    path = f"{OUT}/fig2_3r_workspace.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path}: cells with 4 solutions {int((counts == 4).sum())}, 2 solutions "
          f"{int((counts == 2).sum())}, unreachable {int((counts == 0).sum())}")
    return path


def fig_loop_events() -> str:
    """The SR5 loop's real event table: live tracks per sample, the located crossings, the births,
    and the fibre sizes confirmed by an independent census (numbers from the recorded runs)."""
    samples = np.arange(1, 168)
    counts = np.empty(samples.size)
    for value, lo, hi in ((10, 1, 1), (8, 2, 31), (6, 32, 45), (2, 46, 122),
                          (4, 123, 127), (8, 128, 165), (10, 166, 167)):
        counts[(samples >= lo) & (samples <= hi)] = value
    crossings = [1.417569, 31.269323, 45.384651, 45.394843]
    births = [123, 128, 166]
    census = {0: 10, 1: 10, 2: 8, 31: 8, 32: 6, 45: 6, 46: 2, 123: 4, 124: 4, 128: 8, 129: 8,
              166: 10, 167: 10}
    fig, ax = plt.subplots(figsize=(9.0, 3.8), dpi=140)
    ax.step(samples, counts, where="post", color="#1f77b4", linewidth=1.6,
            label="live tracks (repaired tracker: box guard + duplicate pruning)")
    for value, lo, hi in ((10, 1, 1), (8, 2, 31), (6, 32, 45), (2, 46, 122),
                          (4, 123, 127), (8, 128, 165), (10, 166, 167)):
        ax.hlines(value, lo, hi, color="#1f77b4", linewidth=1.6)
    for index, value in census.items():
        ax.plot([index], [value], "o", color="black", markersize=3.6, zorder=5)
    for index in crossings:
        ax.axvline(index, color="crimson", linestyle="--", linewidth=1.0)
    for k, index in enumerate(births):
        ax.annotate(f"birth +{2 if k != 1 else 4}", (index, 11.3), color="green", fontsize=8,
                    ha="center", xytext=(-14 + 28 * (k % 2), 0), textcoords="offset points")
    ax.plot([], [], "o", color="black", markersize=3.6,
            label="independent census (fresh seeds, polished roots)")
    ax.plot([], [], "--", color="crimson", label="crossing located in task space")
    ax.set_xlabel("sample along the prescribed Cartesian loop")
    ax.set_ylabel("real solutions carried")
    ax.set_ylim(0, 12.5)
    ax.set_title("SR5: the loop's real event table (3 deaths -2/-2/-4, 3 births +2/+4/+2, 8 die = 8 born)")
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    path = f"{OUT}/fig3_loop_events.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path}: counts {counts[0]:.0f} -> {counts[-1]:.0f}, crossings {crossings}, "
          f"births {births}, census points {len(census)}")
    return path


def fig_arcs() -> str:
    """A->B in numbers: the fraction of the loop's Cartesian length each posture can track."""
    fraction = {0: 100.0, 1: 23.2, 2: 30.8, 3: 0.9, 4: 30.8, 5: 30.8, 6: 100.0, 7: 23.2, 8: 30.8,
                9: 0.9}
    ends = {1: "31.27", 7: "31.27", 2: "45.39", 5: "45.39", 4: "45.38", 8: "45.38",
            3: "1.42", 9: "1.42"}
    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=140)
    names = list(fraction)
    values = [fraction[k] for k in names]
    colours = ["#2ca02c" if v > 90 else "#1f77b4" if v > 25 else "#d62728" for v in values]
    bars = ax.bar([str(k) for k in names], values, color=colours)
    for bar, k in zip(bars, names):
        text = "whole loop" if k not in ends else f"ends at {ends[k]}"
        ax.annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.set_xlabel("posture (IK solution index at the starting pose)")
    ax.set_ylabel("% of the loop's Cartesian length")
    ax.set_title("One prescribed loop, ten postures, ten different feasible arcs")
    fig.tight_layout()
    path = f"{OUT}/fig4_feasible_arcs.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path}: fractions {values}")
    return path


def fig_components() -> str:
    """The transposition graph's component structure (11 edges): two 6-element components, four
    isolated roots.  Drawn as clusters because the edge list itself lives in the records."""
    fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=140)
    groups = [("component 1 (closure 720 = |S6|)", [0, 7, 10, 11, 12, 13], 1.0),
              ("component 2 (closure 720 = |S6|)", [1, 5, 8, 9, 14, 15], 4.6),
              ("isolated roots", [2, 3, 4, 6], 8.2)]
    for label, members, x0 in groups:
        for k, member in enumerate(members):
            angle = 2 * np.pi * k / max(len(members), 1)
            x = x0 + 0.75 * np.cos(angle)
            y = 1.0 + 0.75 * np.sin(angle)
            ax.plot([x], [y], "o", markersize=15, color="#1f77b4" if len(members) > 1 else "#999999")
            ax.annotate(str(member), (x, y), color="white", fontsize=7, ha="center", va="center")
        ax.annotate(label, (x0, 0.05), ha="center", fontsize=8)
        if len(members) > 1:
            for k in range(len(members)):
                a = 2 * np.pi * k / len(members)
                b = 2 * np.pi * ((k + 1) % len(members)) / len(members)
                ax.plot([x0 + 0.75 * np.cos(a), x0 + 0.75 * np.cos(b)],
                        [1.0 + 0.75 * np.sin(a), 1.0 + 0.75 * np.sin(b)], color="#cccccc",
                        linewidth=0.8, zorder=0)
    ax.set_xlim(0, 9.2)
    ax.set_ylim(-0.3, 2.2)
    ax.axis("off")
    ax.set_title("Transposition graph: two 6-root components each generate S6 (11 edges)")
    fig.tight_layout()
    path = f"{OUT}/fig5_components.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"{path}: components [6, 6, 1, 1, 1, 1]")
    return path


def main() -> int:
    """Draw the requested figures (all of them by default) and report what each contains."""
    import argparse

    makers = {"aspects": fig_aspects, "workspace": fig_workspace, "events": fig_loop_events,
              "arcs": fig_arcs, "components": fig_components}
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", choices=sorted(makers), default=None,
                        help="draw only these figures (default: all)")
    args = parser.parse_args()
    for name in (args.only or sorted(makers)):
        makers[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
