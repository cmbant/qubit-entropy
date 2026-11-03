"""Cell certificate for G = delta - C* d^2 > 0 on a box of the critical chart (see critical_cells.py).

A cell is discharged when one of the following certain comparisons holds:
 (a) G_mu >= G_mu(c) - sum_k |g_k| h_k - (1/2) sum_kl sup|H_kl| h_k h_l  >  0 for a convex combination
     G_mu = sum_P mu_P G_P of plane functions (G >= G_mu pointwise; the Taylor remainder uses the interval Hessian of
     G_mu over the whole box; c, h = centre and half-widths);
 (b) d^2 <= 1/400 on the whole box (then delta >= (3/10) d^2 > C* d^2 by Theorem thm:fqe-orbit-stability);
 (c) the box lies in a local cone box of an equality point (handled by verify_critical_local_cone.py).
Every sign decision is a certain ball comparison.  The float quantities `fmax`/`dsmall` (and critical_float.py in the driver) only
decide whether a rule is TRIED; they never certify anything.
"""
from fractions import Fraction
from flint import arb
from critical_cells import (T_jets, delta_part, delta_eval, plane_eval, plane_mats, unsafe_set, NPL, PAIRS6, CSTAR, ZERO, EMAX,
                            RMAT, LN2)

EIGHTH = Fraction(1, 400)
TAU_MAX = Fraction(37, 100)          # the chart is covered where |w|^2 <= TAU_MAX (the rest: analytic, tau >= 0.358786; TC-6 checks tau >= 0.36; the driver skips only boxes with min |w|^2 > 0.37+1e-9)


def to_arb(x):
    """exact ball for a float (dyadic) or a Fraction."""
    if isinstance(x, float):
        return arb(x)
    return arb(Fraction(x).numerator)/arb(Fraction(x).denominator)


def box_balls(lo, hi):
    return [to_arb(a).union(to_arb(b)) for a, b in zip(lo, hi)]


def centre_halfwidth(lo, hi):
    """centre and half-widths, exactly: box coordinates are dyadic floats (<= 53-bit mantissa), checked by the assertion that the
    float centre and half-width equal the exact rational ones."""
    if isinstance(lo[0], float):
        c, h = [(a+b)/2 for a, b in zip(lo, hi)], [(b-a)/2 for a, b in zip(lo, hi)]
        assert all(Fraction(x) == (Fraction(a)+Fraction(b))/2 for x, a, b in zip(c, lo, hi)) and \
            all(Fraction(x) == (Fraction(b)-Fraction(a))/2 for x, a, b in zip(h, lo, hi)), 'box centre/half-width not exact'
        return c, h
    return [(Fraction(a)+Fraction(b))/2 for a, b in zip(lo, hi)], [(Fraction(b)-Fraction(a))/2 for a, b in zip(lo, hi)]


def outside_disc(lo, hi):
    m = Fraction(0)
    for i in (4, 5):
        a, b = Fraction(lo[i]), Fraction(hi[i])
        mn = Fraction(0) if a <= 0 <= b else min(abs(a), abs(b))
        m += mn*mn
    return m > TAU_MAX


def combos(vals, gaps=(0.08, 0.08, 0.08)):
    """candidate index sets: best plane alone, then uniform averages over the best k planes (k=2,3,4) when their centre
    values lie within `gaps`.  vals: list of (float value, plane index)."""
    order = sorted(vals, key=lambda t: -t[0])
    out = [[order[0][1]]]
    for k in (2, 3, 4):
        if order[0][0]-order[k-1][0] < gaps[k-2]:
            out.append([order[i][1] for i in range(k)])
    return out


def hd_matrix(mv):
    """Hessian of d^2 = 2 - 2 sqrt(F) in the entries (m11, m22, m12) of the 2x2 matrix m, F = (m11+m22)/2 + sqrt(det m),
    evaluated on the interval vector mv; None if det m or F may vanish."""
    m11, m22, m12 = mv
    D = m11*m22-m12*m12
    if not D.lower() > 0:
        return None
    sD = D.sqrt()
    F = (m11+m22)/2+sD
    if not F.lower() > 0:
        return None
    sF = F.sqrt()
    Da = [m22, m11, -2*m12]
    Dab = [[ZERO, arb(1), ZERO], [arb(1), ZERO, ZERO], [ZERO, ZERO, -arb(2)]]
    Fa = [arb(1)/2+m22/(2*sD), arb(1)/2+m11/(2*sD), -m12/sD]
    return [[-(Dab[a][b]/(2*sD)-Da[a]*Da[b]/(4*D*sD))/sF+Fa[a]*Fa[b]/(2*F*sF) for b in range(3)] for a in range(3)]


def test_cell(lo, hi, sector, stats=None):
    """Return a string tag ('G', 'HS') if the cell is certified, else None.

    Bound used for a convex combination G_mu = sum mu_P G_P of plane functions on the box (centre c, half-widths h):
      delta >= delta_c + D.(t(y)-t(c))          (tangent in the twelve t_ji: delta is convex in t; unsafe rays natural),
      d_P^2 <= d_P^2(m_c) + a_P.(m(y)-m_c) + (1/2) (m(y)-m_c)^T H (m(y)-m_c)   (Taylor in m_P with H enclosed over m-box),
    with t(y), m(y) expanded to second order in y with the interval second derivatives of the nine T entries only."""
    c, h = centre_halfwidth(lo, hi)
    yb = box_balls(lo, hi)
    TB0 = T_jets(yb, sector, 0)
    lam0 = delta_part(TB0, 0)[3]                  # natural enclosure of the probabilities over the box (used only for `pen`)
    yc = [to_arb(x) for x in c]
    TJ = T_jets(yc, sector, 1)
    dp = delta_part(TJ, 1)
    # tangent rays: those with lambda_i(c) > 0 at the CENTRE.  The tangent inequality of the convex delta holds on the whole
    # domain t_i >= -1 (every chart point is a state, so lambda_i(y) >= 0 on the box); only a ray with lambda_i(c) not
    # certified positive is left out and bounded by pen over the box enclosure lam0.
    unsafe = tuple(unsafe_set(dp[3]))
    dl = delta_eval(dp, 1, unsafe, lam0)
    if dl is None:
        return None
    pl = []
    MMc = plane_mats(TJ, 1)
    for p in range(NPL):
        r = plane_eval(p, TJ, dl, 1, MMc)
        if r is not None:
            pl.append((p, r))
    if not pl:
        return None
    harb = [to_arb(x) for x in h]
    vals = [(float(r[0].mid()), p) for p, r in pl]
    byp = dict(pl)
    cand = []
    for S in combos(vals):
        mu = arb(1)/len(S)
        g = [sum((byp[p][1][k] for p in S), ZERO)*mu for k in range(6)]
        Gc = sum((byp[p][0] for p in S), ZERO)*mu
        lb1 = Gc-sum((g[k].abs_upper()*harb[k] for k in range(6)), ZERO)
        if lb1 > 0:
            cand.append((S, lb1))
    fmax = 0.0
    for p in range(NPL):                       # float centre values of F_P: only decides whether the HS rule is tried
        mf = [float(MMc[3*p+q, 0].mid())+(0.25 if q < 2 else 0.0) for q in range(3)]
        fmax = max(fmax, 0.5*(mf[0]+mf[1])+max(mf[0]*mf[1]-mf[2]**2, 0.0)**0.5)
    dsmall = 2-2*fmax**0.5 < 0.012
    if not cand and not dsmall:
        return None
    TB = T_jets(yb, sector, 2)
    MMb = plane_mats(TB, 2)
    if cand:
        t, tg, th, lam = dp
        safe = [i for i in range(12) if i not in unsafe]
        Dd = {i: (1+lam[i].log())/(12*LN2) for i in safe}
        E = [sum((Dd[i]*RMAT[i, n] for i in safe), ZERO) for n in range(9)]
        remd = ZERO
        for n_, (a, b) in enumerate(PAIRS6):
            Th_ab = sum((E[n]*TB[2][n][n_] for n in range(9)), ZERO)
            remd = remd+(1 if a == b else 2)*Th_ab.abs_upper()*harb[a]*harb[b]
        remd = remd/2
        remP = {}
        need = sorted({p for S, _ in cand for p in S})
        for p in need:
            # d^2 pieces for plane p
            m_c = [MMc[3*p+q, 0]+(arb(1)/4 if q < 2 else ZERO) for q in range(3)]
            Dc = m_c[0]*m_c[1]-m_c[2]*m_c[2]
            sDc = Dc.sqrt()
            Fc = (m_c[0]+m_c[1])/2+sDc
            Fm = [arb(1)/2+m_c[1]/(2*sDc), arb(1)/2+m_c[0]/(2*sDc), -m_c[2]/sDc]
            a_ = [-x/Fc.sqrt() for x in Fm]
            rr = []
            for q in range(3):
                lin_q = sum((MMc[3*p+q, 1+k].abs_upper()*harb[k] for k in range(6)), ZERO)
                sec_q = ZERO
                for n_, (a, b) in enumerate(PAIRS6):
                    sec_q = sec_q+(1 if a == b else 2)*MMb[3*p+q, 7+n_].abs_upper()*harb[a]*harb[b]
                rr.append(lin_q+sec_q/2)               # |m_q(y) - m_q(c)| <= rr_q on the box
            # Bregman remainder of the convex function d^2(m): sup over the box |dm_q| <= rr_q is attained at a corner
            t2 = None
            d2c = 2-2*Fc.sqrt()
            for sg_ in ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)):
                mm = [m_c[q]+sg_[q]*rr[q] for q in range(3)]
                Dm = mm[0]*mm[1]-mm[2]*mm[2]
                Fq = (mm[0]+mm[1])/2+Dm.sqrt() if Dm.lower() > 0 else None
                if Fq is None or not Fq.lower() > 0:
                    t2 = None
                    break
                u = (2-2*Fq.sqrt()-d2c-sum((a_[q]*sg_[q]*rr[q] for q in range(3)), ZERO)).upper()
                t2 = u if t2 is None or u > t2 else t2
            if t2 is None:
                remP[p] = None
                continue
            t1 = ZERO
            for n_, (a, b) in enumerate(PAIRS6):
                Phi = sum((a_[q]*MMb[3*p+q, 7+n_] for q in range(3)), ZERO)
                t1 = t1+(1 if a == b else 2)*Phi.abs_upper()*harb[a]*harb[b]
            remP[p] = t1/2+t2
        for S, lb1 in cand:
            if any(remP[p] is None for p in S):
                continue
            mu = arb(1)/len(S)
            rem = remd+CSTAR*sum((remP[p] for p in S), ZERO)*mu
            if lb1-rem > 0:
                if stats is not None:
                    stats['lb'] = float((lb1-rem).lower())       # certified lower bound of G on the box (audit only)
                return 'G'
    if dsmall:
        # d^2 <= 1/400 on the box: lower bound of F_P for the best plane from the natural enclosure of m_P over the box
        best = max(pl, key=lambda t: float(t[1][0].mid()))[0]
        m = [MMb[3*best+q, 0]+(arb(1)/4 if q < 2 else ZERO) for q in range(3)]
        D = m[0]*m[1]-m[2]*m[2]
        if D.lower() > 0:
            F = (m[0]+m[1])/2+D.sqrt()
            d2hi = 2-2*F.lower().sqrt() if F.lower() > 0 else None
            if d2hi is not None and d2hi < arb(EIGHTH.numerator)/arb(EIGHTH.denominator):      # certain ball comparison
                return 'HS'
    return None


def split(lo, hi, weights=None):
    w = [Fraction(b)-Fraction(a) for a, b in zip(lo, hi)]
    if weights:
        w = [x*wt for x, wt in zip(w, weights)]
    i = max(range(len(w)), key=lambda k: w[k])
    m = (Fraction(lo[i])+Fraction(hi[i]))/2
    return i, m
