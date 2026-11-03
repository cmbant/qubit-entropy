"""Floating-point (numpy) twin of critical_cells.py, used ONLY to steer the bisection (which cells to try first).

Nothing here is part of a proof: every cell that the covering discharges is certified by the ball arithmetic of
critical_cover_lib.test_cell.  The float code decides which cells are not worth an arb test (their first-order bound is
clearly negative) and is cross-checked against the arb code in verify_critical_chart.py.
"""
import numpy as np
from critical_cells import RAYS, MP, EMAX, CSTAR, NPL

CF = float(CSTAR.mid())
EMAXF = float(EMAX.mid())
RA = np.array([[r[k][0] for k in range(3)] for r in RAYS])           # (12,3) row index a
RB = np.array([[r[k][1] for k in range(3)] for r in RAYS])
RS = np.array([[r[k][2] for k in range(3)] for r in RAYS], float)
MPF = np.array([[[float(MP[p][n][c].mid()) for c in range(3)] for n in range(9)] for p in range(NPL)])   # (16,9,3)


def side_f(P, Q, sg):
    N = np.sqrt(1+P*P+Q*Q)
    r = np.stack([sg/N, P/N, Q/N], -1)
    nf = np.sqrt(1+P*P)
    f = np.stack([P/nf, -sg/nf, np.zeros_like(P)], -1)
    return r, f, np.cross(r, f)


def T_f(y, sector):
    r, xa, ya = side_f(y[..., 0], y[..., 1], np.ones_like(y[..., 0]))
    s, xb, yb = side_f(y[..., 2], y[..., 3], np.full_like(y[..., 0], float(sector)))
    w1 = y[..., 4, None, None]
    w2 = y[..., 5, None, None]
    return (r[..., :, None]*s[..., None, :]+w1*(xa[..., :, None]*xb[..., None, :]-ya[..., :, None]*yb[..., None, :])
            + w2*(xa[..., :, None]*yb[..., None, :]+ya[..., :, None]*xb[..., None, :]))


def Gp_f(y, sector, with_aux=False):
    T = T_f(y, sector)                                   # (...,3,3)
    t = (T[..., RA, RB]*RS).sum(-1)                      # (...,12)
    lam = (1+t)/4
    l = np.clip(lam, 1e-300, None)
    dl = EMAXF+(l*np.log2(l)).sum(-1)/3
    Tf = T.reshape(T.shape[:-2]+(9,))
    m = 0.25*np.einsum('...n,pnc->...pc', Tf, MPF)       # (...,16,3)
    m[..., 0] += 0.25
    m[..., 1] += 0.25
    D = np.clip(m[..., 0]*m[..., 1]-m[..., 2]**2, 0, None)
    F = 0.5*(m[..., 0]+m[..., 1])+np.sqrt(D)
    G = dl[..., None]-CF*(2-2*np.sqrt(F))
    if with_aux:
        return G, lam, F
    return G


def derivs(yc, sector, step):
    N = len(yc)
    f0 = Gp_f(yc, sector)
    g = np.zeros((N, NPL, 6))
    H = np.zeros((N, NPL, 6, 6))
    for a in range(6):
        e = np.zeros((N, 6))
        e[:, a] = step[:, a]
        fp = Gp_f(yc+e, sector)
        fm = Gp_f(yc-e, sector)
        g[:, :, a] = (fp-fm)/(2*step[:, a, None])
        H[:, :, a, a] = (fp-2*f0+fm)/(step[:, a, None]**2)
    for a in range(6):
        for b in range(a):
            ea = np.zeros((N, 6))
            ea[:, a] = step[:, a]
            eb = np.zeros((N, 6))
            eb[:, b] = step[:, b]
            v = (Gp_f(yc+ea+eb, sector)-Gp_f(yc+ea-eb, sector)-Gp_f(yc-ea+eb, sector)+Gp_f(yc-ea-eb, sector)) \
                / (4*step[:, a, None]*step[:, b, None])
            H[:, :, a, b] = v
            H[:, :, b, a] = v
    return f0, g, H


def candidates(lo, hi, sector, infl=2.0):
    """float heuristic: True where a second-order bound (Hessian inflated by `infl`) of some uniform combination of the
    best planes is positive, or where the centre is within d^2 < 0.02 of an orbit point (HS rule)."""
    hw = (hi-lo)/2
    c = (hi+lo)/2
    step = np.maximum(hw*0.5, 1e-6)
    f0, g, H = derivs(c, sector, step)
    N = len(c)
    order = np.argsort(-f0, axis=1)[:, :4]
    rows = np.arange(N)[:, None]
    top = f0[rows, order]
    best = np.full(N, -1e9)
    cub = 0.2*(hw.sum(1)**3/6)
    for k in (1, 2, 3, 4):
        sel = order[:, :k]
        fk = f0[rows, sel].mean(1)
        gk = g[rows, sel].mean(1)
        Hk = H[rows, sel].mean(1)
        lin = (np.abs(gk)*hw).sum(1)
        quad = 0.5*infl*np.einsum('nab,na,nb->n', np.abs(Hk), hw, hw)
        lb = fk-lin-quad-cub
        ok = (top[:, 0]-top[:, k-1] < 0.08) | (k == 1)
        best = np.maximum(best, np.where(ok, lb, -1e9))
    G, lam, F = Gp_f(c, sector, True)
    d2c = 2-2*np.sqrt(F.max(1))
    return (best > -1e-9) | (d2c < 0.03)
