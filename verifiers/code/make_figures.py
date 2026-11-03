#!/usr/bin/env python3
"""Deterministic figures of the paper, written to ../figures/ as PDF (../paper/figures/ in the companion).

Formulas: entropy_maximum.tex (Hermite cubic), spectral_corollaries.tex
(HS/cluster spectra), and renyi_above_one.tex (crossing).
Figure sizes target an A4 text width of 164 mm, at the LaTeX inclusion fractions.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / 'paper' / 'figures' if (ROOT.parent / 'paper').is_dir() else ROOT / 'figures'  # figures go to ../paper/figures/ when that directory exists
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 8,
    'axes.linewidth': 0.65, 'lines.linewidth': 1.3, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'figure.dpi': 150, 'pdf.fonttype': 42,
})
BLUE, ORANGE, GREY = '#0072B2', '#D55E00', '0.45'  # Okabe--Ito; also use dashes/markers.
WIDTH = 164 / 25.4


def save(fig, name):
    # Fixed page sizes preserve font sizes at the manuscript's inclusion widths.
    fig.savefig(OUT / f'{name}.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    print(f'wrote figures/{name}.pdf')


# Hermite majorant: h and f agree in value and derivative at both HS eigenvalues.
a, b = 18*(np.log(3) - 1), 18 - 19.5*np.log(3)


def f(x):
    return -x*np.log(np.where(x > 0, x, 1))


c = -np.log(.5) - 1 - 3*a*.25 - b
d = f(.5) - a/8 - b/4 - c/2


def h(x):
    return a*x**3 + b*x**2 + c*x + d


nodes = np.array([1/6, 1/2])
assert np.allclose(h(nodes), f(nodes), atol=1e-14)
assert np.allclose(3*a*nodes**2 + 2*b*nodes + c, -np.log(nodes)-1, atol=1e-14)
x = np.unique(np.r_[np.linspace(0, 1, 801), nodes])
fig, ax = plt.subplots(figsize=(WIDTH*.5, 2.6), layout='constrained')
ax.plot(x, f(x), color='0.1', label=r'$f(x)=-x\ln x$')
ax.plot(x, h(x), color=BLUE, ls='--', label=r'cubic majorant $h$')
ax.plot(nodes, f(nodes), 'o', color=BLUE, ms=3.5)
ax.set(xlabel=r'eigenvalue $x$', ylabel='entropy (nats)', ylim=(-.01, .48), xlim=(0, 1))
ax.set_xticks([0, 1/6, 1/2, 1], ['0', r'$1/6$', r'$1/2$', '1'])
ax.legend(loc='upper right', frameon=False)
# Inset: the gap on [0.1, 0.6], where it is O(1e-3) and the two double zeros are visible.
gap = ax.inset_axes([.3, .08, .44, .36])
xs = np.unique(np.r_[np.linspace(.1, .6, 401), nodes])
gap.fill_between(xs, 0, 1e3*np.maximum(h(xs)-f(xs), 0), color='0.92')
gap.plot(xs, 1e3*(h(xs)-f(xs)), color=BLUE)
gap.plot(nodes, [0, 0], 'o', color=BLUE, ms=3, clip_on=False)
for xi in nodes:
    gap.axvline(xi, color=GREY, lw=.6, ls=':')
gap.set_xlim(.1, .6)
gap.set_ylim(0, None)
gap.set_xticks([1/6, 1/2], [r'$1/6$', r'$1/2$'])
gap.tick_params(labelsize=8, length=2)
gap.set_title(r'gap $10^3(h-f)$', fontsize=8, pad=2)
save(fig, 'fig_majorant')


# Continuous Rényi entropy, including q=1 without cancellation or a hole.
HS = np.array([1/2, 1/6, 1/6, 1/6])


def entropy(q):
    spectrum = HS
    r = np.asarray(q, dtype=float) - 1
    delta = (spectrum*np.expm1(r[..., None]*np.log(spectrum))).sum(axis=-1)
    numerator = -np.log1p(delta)/np.log(2)
    limit = -(spectrum*np.log(spectrum)).sum() / np.log(2)
    return np.divide(numerator, r, out=np.full_like(r, limit), where=r != 0)


qs = brentq(lambda q: entropy(q)-5/3, 1.5, 1.6)
assert 1.587 < qs < 1.589  # Existing exact bracket: RA-4 and its committed log.
q = np.unique(np.r_[np.linspace(0, 3, 601), qs])


fig, ax = plt.subplots(figsize=(WIDTH*.8, 2.75), layout='constrained')
# Faint regime shading and labels mark which state attains the maximum on each side of q_*.
ax.axvspan(0, qs, color=BLUE, alpha=.06, lw=0, zorder=0)
ax.axvspan(qs, 3, color=ORANGE, alpha=.06, lw=0, zorder=0)
ax.text(qs/2, 1.47, r'$M_4$ maximal', color=BLUE, ha='center', va='center', fontsize=8.5)
ax.text((qs+3)/2, 1.725, 'cluster maximal', color=ORANGE, ha='center', va='center', fontsize=8.5)
ax.plot(q, entropy(q), color=BLUE, label=r'$M_4$: $h_q$')
ax.axhline(5/3, color=ORANGE, ls='--', label=r'cluster: $5/3$')
ax.axvline(qs, color=GREY, lw=.7, ls=':')
ax.plot(qs, 5/3, 'o', color='0.15', ms=3, zorder=3)
ax.annotate(rf'$q_*\approx{qs:.3f}$', (qs, 1.885), xytext=(5, 0),
            textcoords='offset points', fontsize=8)
ax.set(xlabel=r'order $q$', ylabel='mean Rényi entropy (bits)',
       xlim=(0, 3), ylim=(1.4, 2.04))
ax.legend(loc='upper right', frameon=False)
save(fig, 'fig_renyi')


# Reproducible vector comparison of all three HS/cluster cut spectra.
from make_fig_cuts import make_cuts_figure
make_cuts_figure()

from make_fig_icosahedron import make_icosahedron_figure
make_icosahedron_figure()
