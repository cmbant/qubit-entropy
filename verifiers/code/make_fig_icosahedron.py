#!/usr/bin/env python3
"""Icosahedral axis signs and exact triple spectra of Psi_ico (IC-6).

Run: python3 code/make_fig_icosahedron.py
Writes the vector PDF figures/fig_icosahedron.pdf. The six labelled dots
select oriented representatives of antipodal axes; s is a separate qubit.
The dashed chords encode signs, not physical edges or couplings.
"""
from pathlib import Path
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patheffects as pe
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "paper" / "figures" if (ROOT.parent / "paper").is_dir() else ROOT / "figures"  # ../paper/figures/ in the companion
WIDTH = 164 / 25.4
PHI = (1 + np.sqrt(5)) / 2
V = np.array([
    [0, 1, PHI], [0, -1, PHI], [0, 1, -PHI], [0, -1, -PHI],
    [1, PHI, 0], [-1, PHI, 0], [1, -PHI, 0], [-1, -PHI, 0],
    [PHI, 0, 1], [PHI, 0, -1], [-PHI, 0, 1], [-PHI, 0, -1],
], dtype=float)
V /= np.linalg.norm(V, axis=1)[:, None]

DISTS = np.linalg.norm(V[:, None] - V[None], axis=2)
SIDE = np.min(DISTS[DISTS > 1e-12])
EDGES = {tuple((i, j)) for i, j in combinations(range(12), 2)
         if abs(DISTS[i, j] - SIDE) < 1e-9}
FACES = [f for f in combinations(range(12), 3)
         if all(tuple(sorted(e)) in EDGES for e in combinations(f, 2))]
assert len(EDGES) == 30 and len(FACES) == 20

# The oriented representative u_0 is at vertex 0. The five neighbours
# carry the cyclic *geometric* order 1,2,5,3,4, giving the manuscript's
# negative-sign pentagram 1--3--2--4--5--1.
I0 = 0
U0 = V[I0]
NEIGHBOURS = sorted({j if i == I0 else i for i, j in EDGES if i == I0 or j == I0})
REF = np.array([1.0, 0.0, 0.0])
EX = REF - np.dot(REF, U0) * U0
EX /= np.linalg.norm(EX)
EY = np.cross(U0, EX)
ORDER = sorted(NEIGHBOURS, key=lambda j: np.arctan2(np.dot(V[j], EY), np.dot(V[j], EX)))
LABELED_VERTEX = {0: I0, **dict(zip((1, 2, 5, 3, 4), ORDER))}
NEGATIVE_PAIRS = [(1, 3), (2, 3), (2, 4), (4, 5), (1, 5)]

for i, j in combinations(range(6), 2):
    dot = float(np.dot(V[LABELED_VERTEX[i]], V[LABELED_VERTEX[j]]))
    expected = -1 if ((i, j) in NEGATIVE_PAIRS or (j, i) in NEGATIVE_PAIRS) else 1
    assert np.isclose(dot, expected / np.sqrt(5), atol=1e-12)

# Exact eigenvalues for the icosahedral state at x=37/524.
SPECTRUM_ORDINARY = np.array([41/262]*2 + [15/131]*6)
SPECTRUM_SPECIAL = np.array([30/131] + [15/131]*5 + [13/131]*2)
assert np.isclose(SPECTRUM_ORDINARY.sum(), 1)
assert np.isclose(SPECTRUM_SPECIAL.sum(), 1)
assert np.isclose((20*np.sum(SPECTRUM_ORDINARY**2) +
                   15*np.sum(SPECTRUM_SPECIAL**2))/35, 121/917)

BLUE = "#2F79B5"
GREEN = "#3A8F3E"
RED = "#B55D6A"
ORANGE = "#D55E00"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "font.size": 8.5,
    "axes.titlesize": 9, "axes.labelsize": 8.5, "pdf.fonttype": 42,
})


def camera():
    # Oblique view with axis 0 slightly in front, not exactly centred.
    view = U0 + 0.32 * V[ORDER[0]] + 0.22 * V[ORDER[1]]
    view /= np.linalg.norm(view)
    x_cam = V[ORDER[3]] - np.dot(V[ORDER[3]], view) * view
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(view, x_cam)
    return view, x_cam, y_cam


def draw_icosahedron(ax):
    view, xc, yc = camera()
    xy = np.column_stack((V @ xc, V @ yc))
    depth = V @ view
    light = np.array([0.55, 0.25, 1.0])
    light /= np.linalg.norm(light)

    # Draw only front-facing faces, with face-by-face directional shading.
    front_faces = []
    for f in FACES:
        triangle = V[list(f)]
        n = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        n /= np.linalg.norm(n)
        if np.dot(n, triangle.mean(axis=0)) < 0:
            n *= -1
        if np.dot(n, view) <= 0:
            continue
        shade = 0.55 + 0.35 * max(np.dot(n, light), 0)
        front_faces.append((depth[list(f)].mean(), xy[list(f)], shade))
    for _, poly, shade in sorted(front_faces, key=lambda item: item[0]):
        face = (0.70 * shade, 0.82 * shade, 0.98 * shade)
        edge = (0.30 * shade, 0.45 * shade, 0.68 * shade)
        ax.add_patch(Polygon(poly, closed=True, facecolor=face,
                             edgecolor=edge, lw=0.8, zorder=2))

    # Dashed red chords encode the five -1 signs (they aren't solid edges).
    for a, b in NEGATIVE_PAIRS:
        p, q = xy[LABELED_VERTEX[a]], xy[LABELED_VERTEX[b]]
        ax.plot((p[0], q[0]), (p[1], q[1]), color=RED, lw=1.35,
                linestyle=(0, (3.4, 2.2)), zorder=4)

    c = xy[LABELED_VERTEX[0]]
    for i in range(6):
        p = xy[LABELED_VERTEX[i]]
        ax.plot(p[0], p[1], "o", markersize=5.6, color=ORANGE, zorder=5)
        if i == 0:
            # White halo keeps the central label legible over the face edges.
            ax.annotate("0", p, xytext=(6, -3), textcoords="offset points",
                        va="top", ha="left", fontsize=10, zorder=6,
                        path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
        else:
            direction = p - c
            direction /= np.linalg.norm(direction)
            label_pt = p + (.15 if i in (1, 2, 5) else .092) * direction
            ax.text(label_pt[0], label_pt[1], str(i), fontsize=10,
                    ha="center", va="center", zorder=6)

    xmin, ymin = xy.min(axis=0) - 0.2
    xmax, ymax = xy.max(axis=0) + 0.2
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Icosahedron and six vertex axes", pad=7)


def draw_spectrum(ax, values, title, groups, show_ame=False):
    ax.bar(range(1, 9), values, width=.68, color=BLUE)
    ax.axhline(1/8, color=GREEN, linewidth=1.2, linestyle="--")
    if show_ame:
        ax.text(8.7, 1/8 + .006, "maximally mixed", fontsize=8,
                color=GREEN, ha="right", va="bottom")
    ax.set(xlim=(.4, 8.95), ylim=(0, .255))
    ax.set_yticks([0, 1/8, 1/4], ["0", r"$1/8$", r"$1/4$"])
    ax.set_xticks([])
    ax.set_ylabel("Eigenvalue")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, loc="left", pad=5)
    # Each eigenvalue label is centred under its run of equal bars.
    start = 1
    for label, count in groups:
        assert np.allclose(values[start-1:start-1+count], values[start-1])
        ax.text(start + (count-1)/2, -.045, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5)
        start += count
    assert start == 9


def make_figure():
    # Explicit margins reserve independent footer space below the left panel.
    fig = plt.figure(figsize=(WIDTH, 3.65), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.04, 1.46],
                          left=.055, right=.977, top=.895,
                          bottom=.24, wspace=.14)
    ax = fig.add_subplot(gs[0])
    draw_icosahedron(ax)

    right = gs[1].subgridspec(2, 1, hspace=.67)
    upper = fig.add_subplot(right[0])
    lower = fig.add_subplot(right[1])
    draw_spectrum(upper, SPECTRUM_ORDINARY,
                  r"20 triples avoiding $s$",
                  [(r"$41/262\ (\times2)$", 2), (r"$15/131\ (\times6)$", 6)], show_ame=True)
    draw_spectrum(lower, SPECTRUM_SPECIAL,
                  r"15 triples containing $s$",
                  [(r"$30/131$", 1), (r"$15/131\ (\times5)$", 5), (r"$13/131\ (\times2)$", 2)])


    # Sign legend below the geometry; the caption explains the axis labels and s.
    leftbox = ax.get_position()
    fig.text((leftbox.x0+leftbox.x1)/2, leftbox.y0-.028,
             r"Dashed chords: $s_{ij}=-1$", ha="center", va="top", fontsize=8)
    return fig


def make_icosahedron_figure():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    fig.savefig(OUT / "fig_icosahedron.pdf", bbox_inches="tight", pad_inches=.10,
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print("wrote figures/fig_icosahedron.pdf")


if __name__ == "__main__":
    make_icosahedron_figure()
