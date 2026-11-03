"""Local cone lemma at an equality point of the critical chart (second-order Taylor on rays, interval Hessian).

At an equality point y_a (w=0) exactly four planes P are tied (G_P(y_a)=0 = G(y_a)); G = max_P G_P.  In the local
coordinates z (z_i = +-(y_i - y_a,i) >= 0 for the four chart coordinates, z_5, z_6 = w) and the box |z|_inf <= rho,
    G_P(t w) = t g_P.w + (t^2/2) w^T H_P(xi) w,   H_P(xi) in [H_P]  (interval Hessian over the box),
and for weights mu (convex combination of the tied planes) G >= sum_P mu_P G_P.  A cone cell (a piece Omega of the sup-norm
sphere) is discharged if  l_min >= 0  and  l_min + rho*q_min > 0  (or q_min > 0), where
l = sum mu_P g_P.w, q = (1/2) w^T [H_mu] w  over w in Omega; then G(t w) > 0 for all 0 < t <= rho.
The uniform weights have l = 0 identically, because the V4 symmetry of the equality point gives sum_P g_P = 0
(here the enclosure of that sum is only re-checked to contain 0; the exact tie G_P(apex) = 0 of the four planes and the exact
vanishing of the gradient sum are proved symbolically for all eight points by verify_critical_apex_exact.py); weights are written mu = 1/4 + nu, sum nu = 0, so that only
nu enters l.
"""
from fractions import Fraction
from flint import arb
from critical_cells import (T_jets, delta_part, delta_eval, plane_eval, unsafe_set, NPL, PAIRS6, ZERO)
from critical_cover_lib import to_arb, box_balls


def apex_data(sector, apex, rho):
    """tied planes, their z-gradients g[p] (arb 6-list) and z-Hessian enclosures H[p] (21-list) over the core box."""
    sig = [1 if a == 0 else -1 for a in apex]+[1, 1]
    ya = [arb(a) for a in apex]+[arb(0), arb(0)]
    TJ = T_jets(ya, sector, 1)
    dp = delta_part(TJ, 1)
    dl = delta_eval(dp, 1, ())
    vals = [plane_eval(p, TJ, dl, 1) for p in range(NPL)]
    tied = [p for p in range(NPL) if vals[p] is not None and vals[p][0].contains(0) and vals[p][0].rad() < arb('1e-25')]
    others = [p for p in range(NPL) if p not in tied]
    for p in others:
        assert vals[p] is None or vals[p][0].upper() < -arb('0.05'), (p, vals[p][0])
    g = {p: [sig[k]*vals[p][1][k] for k in range(6)] for p in tied}
    lo = [a if a == 0 else 1-rho for a in apex]+[-rho, -rho]
    hi = [rho if a == 0 else 1 for a in apex]+[rho, rho]
    yb = box_balls(lo, hi)
    TB0 = T_jets(yb, sector, 0)
    assert not unsafe_set(delta_part(TB0, 0)[3]), 'a probability may vanish in the core box'
    TB = T_jets(yb, sector, 2)
    dlb = delta_eval(delta_part(TB, 2), 2, ())
    H = {}
    for p in tied:
        r = plane_eval(p, TB, dlb, 2)
        H[p] = [sig[a]*sig[b]*r[2][n] for n, (a, b) in enumerate(PAIRS6)]
    return tied, g, H, vals


def quad_lower(Hmu, om):
    """lower bound of (1/2) w^T H w for H intervals (21 slots) and w = om (list of arb balls)."""
    s = ZERO
    for n, (a, b) in enumerate(PAIRS6):
        f = 1 if a == b else 2
        s = s+f*Hmu[n]*om[a]*om[b]
    return s/2


def cell_ok(g, H, tied, om, nus, rho):
    """try the weight offsets nus (list of 4-tuples nu, sum 0) on the cell om (arb balls); returns (nu, l_min, q_min) if certified,
    else None."""
    for nu in nus:
        mu = [arb(1)/4+arb(Fraction(x).numerator)/arb(Fraction(x).denominator) for x in nu]
        if any(not m.lower() >= 0 for m in mu):
            continue
        ell = ZERO
        for p, x in zip(tied, nu):
            if x != 0:
                ell = ell+(arb(Fraction(x).numerator)/arb(Fraction(x).denominator))*sum((g[p][k]*om[k] for k in range(6)), ZERO)
        Hm = [sum((mu[i]*H[tied[i]][n] for i in range(4)), ZERO) for n in range(21)]
        q = quad_lower(Hm, om)
        lmin = ell.lower()
        qmin = q.lower()
        if lmin >= 0 and (qmin > 0 or lmin+rho*qmin > 0):
            return nu, float(lmin), float(qmin)
    return None


def candidate_nus(g, tied, omc):
    """weight offsets to try on a cell with centre omc (floats): uniform, single planes, graded 'tetrahedral' offsets."""
    a = [float(sum((g[p][k]*omc[k] for k in range(6)), arb(0)).mid()) for p in tied]
    out = [(0, 0, 0, 0)]
    m = max(abs(x) for x in a) or 1.0
    for th in (Fraction(1, 64), Fraction(1, 32), Fraction(1, 16), Fraction(1, 8), Fraction(3, 16), Fraction(1, 4)):
        nu = tuple(Fraction(round(float(th)*x/m*256), 256) for x in a)
        s = sum(nu)
        nu = tuple(list(nu[:3])+[nu[3]-s])
        out.append(nu)
    i = max(range(4), key=lambda k: a[k])
    single = [Fraction(-1, 4)]*4
    single[i] = Fraction(3, 4)
    out.append(tuple(single))
    return out


def cone_cells(sector, apex, rho, maxdepth=60):
    tied, g, H, vals = apex_data(sector, apex, rho)
    rs = arb(Fraction(rho).numerator)/arb(Fraction(rho).denominator)
    summ = [sum((g[p][k] for p in tied), ZERO) for k in range(6)]
    assert all(x.contains(0) and x.rad() < arb('1e-25') for x in summ), 'sum of tied gradients is not 0'
    st = dict(cells=0, depth=0)
    stack = []
    for f in range(6):
        for sgn in ((1,) if f < 4 else (1, -1)):
            lo = [Fraction(0) if i < 4 else Fraction(-1) for i in range(6)]
            hi = [Fraction(1)]*6
            lo[f] = hi[f] = Fraction(sgn)
            stack.append((lo, hi, 0))
    while stack:
        lo, hi, d = stack.pop()
        om = [arb(Fraction(a).numerator)/arb(Fraction(a).denominator) if a == b else
              (arb(Fraction(a).numerator)/arb(Fraction(a).denominator)).union(arb(Fraction(b).numerator)/arb(Fraction(b).denominator))
              for a, b in zip(lo, hi)]
        omc = [float((Fraction(a)+Fraction(b))/2) for a, b in zip(lo, hi)]
        cert = cell_ok(g, H, tied, om, candidate_nus(g, tied, omc), rs)
        if cert is not None:
            st['cells'] += 1
            st['depth'] = max(st['depth'], d)
            st.setdefault('data', []).append((lo, hi, cert))
            continue
        if d >= maxdepth:
            return None, st
        w = [Fraction(b)-Fraction(a) for a, b in zip(lo, hi)]
        i = max(range(6), key=lambda k: w[k])
        m = (Fraction(lo[i])+Fraction(hi[i]))/2
        h1 = list(hi)
        h1[i] = m
        l2 = list(lo)
        l2[i] = m
        stack.append((list(lo), h1, d+1))
        stack.append((l2, list(hi), d+1))
    return True, st
