"""Shared rigorous (python-flint arb, outward rounding) evaluator for the critical-locus sharp coefficient.

Model (critical_sharp_coefficient.tex): a critical state is a real density matrix A = 1/4 I + 1/4 sum_ji t_ji B_ji
(three real bases, twelve rank-one projectors B_ji = h_ji h_ji^T), t_ji = <C^{ji}, T>, where T is the 3x3
correlation matrix of the auxiliary two-qubit vector.  In the chart used here
    T = r s^T + w1 (x_a x_b^T - y_a y_b^T) + w2 (x_a y_b^T + y_a x_b^T),     |w|^2 = tau,
    r=(1,P,Q)/N, x_a=(P,-1,0)/n, y_a=r x x_a;   s=(sg,P',Q')/N', x_b=(P',-sg,0)/n', y_b=s x x_b,
with y=(P,Q,P',Q',w1,w2) and sg=+-1 the sign of s_1 (the parity sector).  For each of the sixteen planes P=V V^T,
    m_P = V^T A V (2x2),  F_P = (1/2) tr m_P + sqrt(det m_P),  F = max_P F_P,  d^2 = 2 - 2 sqrt(F),
    delta = EMAX - E,  E = (1/3) sum_j H(lambda_j.),  lambda_ji = (1+t_ji)/4,
    G_P = delta - C* (2 - 2 sqrt(F_P)),   G = delta - C* d^2 = max_P G_P.
Everything below is exact ball arithmetic: a sign decision is taken only when a ball excludes zero.
"""
from flint import arb, arb_mat, ctx
from critical_side_jets import side_jets0, side_jets1, side_jets2

ctx.prec = 100
S3 = arb(3).sqrt()
LN2 = arb(2).log()
A0 = arb(3).log()/LN2/2
EMAX = 1+A0
_p = (2+arb(2).sqrt())/4
_H2 = -(_p*_p.log()+(1-_p)*(1-_p).log())/LN2
CSTAR = (A0-(1+2*_H2)/3)/(2-((1+1/S3)*(1+(arb(2)/3).sqrt())).sqrt())
CSTAR_HI = CSTAR.upper()
ZERO = arb(0)

# ---- the twelve rays: t_ji = sum sign*T[a][b]  (signed permutation matrices C^{ji}; recomputed from the physical
# magic basis in verify_critical_chart.py) ----------------------------------------------------------------------
_C = {
    (0, 0): [(0, 0, 1), (1, 1, -1), (2, 2, 1)], (0, 1): [(0, 0, 1), (1, 1, 1), (2, 2, -1)],
    (0, 2): [(0, 0, -1), (1, 1, -1), (2, 2, -1)], (0, 3): [(0, 0, -1), (1, 1, 1), (2, 2, 1)],
    (1, 0): [(0, 1, 1), (1, 2, -1), (2, 0, 1)], (1, 1): [(0, 1, -1), (1, 2, -1), (2, 0, -1)],
    (1, 2): [(0, 1, -1), (1, 2, 1), (2, 0, 1)], (1, 3): [(0, 1, 1), (1, 2, 1), (2, 0, -1)],
    (2, 0): [(0, 2, 1), (1, 0, 1), (2, 1, -1)], (2, 1): [(0, 2, -1), (1, 0, -1), (2, 1, -1)],
    (2, 2): [(0, 2, 1), (1, 0, -1), (2, 1, 1)], (2, 3): [(0, 2, -1), (1, 0, 1), (2, 1, 1)],
}
RAYS = [_C[(j, i)] for j in range(3) for i in range(4)]
_H1 = [[1, -1, 1, -1], [-1, 1, 1, -1], [1, 1, 1, 1], [-1, -1, 1, 1]]
_H2M = [[1, -1, -1, -1], [-1, 1, -1, -1], [1, 1, -1, 1], [-1, -1, -1, 1]]
HROWS = [[[arb(1) if a == b else arb(0) for b in range(4)] for a in range(4)],
         [[arb(x)/2 for x in row] for row in _H1], [[arb(x)/2 for x in row] for row in _H2M]]
RAYROWS = [HROWS[j][i] for j in range(3) for i in range(4)]         # same order as RAYS

# ---- the sixteen planes; MP[p][3a+b] = (11,22,12) components of sum_ji C^{ji}_{ab} u_ji u_ji^T,  u_ji = V^T h_ji ---
PLANES = []
for _k in range(4):
    _oth = [i for i in range(4) if i != _k]
    for _sg in ((1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)):
        PLANES.append((_k, _oth, _sg))
NPL = len(PLANES)
MP = []
for (_k, _oth, _sg) in PLANES:
    cons = [[arb(0), arb(0), arb(0)] for _ in range(9)]
    for ray, h in zip(RAYS, RAYROWS):
        u1 = h[_k]
        u2 = (_sg[0]*h[_oth[0]]+_sg[1]*h[_oth[1]]+_sg[2]*h[_oth[2]])/S3
        comp = (u1*u1, u2*u2, u1*u2)
        for (a, b, s) in ray:
            for c in range(3):
                cons[3*a+b][c] = cons[3*a+b][c]+s*comp[c]
    MP.append(cons)

H6 = {}
_n = 0
for _a in range(6):
    for _b in range(_a, 6):
        H6[(_a, _b)] = _n
        H6[(_b, _a)] = _n
        _n += 1
PAIRS6 = [(a, b) for a in range(6) for b in range(a, 6)]     # slot order of the 21 Hessian entries
_S4 = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3, (1, 1): 4, (1, 2): 5, (1, 3): 6, (2, 2): 7, (2, 3): 8, (3, 3): 9}


def _prod15(A, B):
    """A,B are 6-lists (v, d_u, d_v, dd_uu, dd_uv, dd_vv) in disjoint variable pairs: product jet in 4 variables:
    [v, d0..d3, dd(00,01,02,03,11,12,13,22,23,33)]."""
    return [A[0]*B[0], A[1]*B[0], A[2]*B[0], A[0]*B[1], A[0]*B[2],
            A[3]*B[0], A[4]*B[0], A[1]*B[1], A[1]*B[2], A[5]*B[0], A[2]*B[1], A[2]*B[2],
            A[0]*B[3], A[0]*B[4], A[0]*B[5]]


def _prod5(A, B):
    return [A[0]*B[0], A[1]*B[0], A[2]*B[0], A[0]*B[1], A[0]*B[2]]


def _sub(X, Y):
    return [x-y for x, y in zip(X, Y)]


def _addl(X, Y):
    return [x+y for x, y in zip(X, Y)]


def _sqrt(x):
    return x.sqrt()


def side(P, Q, sg, order=2):
    """nine chart vectors r(0-2), f(3-5), g(6-8) with derivatives: (v,dP,dQ,dPP,dPQ,dQQ) truncated by `order`."""
    fn = (side_jets0, side_jets1, side_jets2)[order]
    k = (1, 3, 6)[order]
    o = fn(P, Q, arb(sg), _sqrt)
    return [o[k*c:k*c+k] for c in range(9)]


def T_jets(y, sector, order):
    """Jets of the nine entries T[a][b] (index 3a+b): value, grad6 and (order 2) hess21 w.r.t. y=(P,Q,P',Q',w1,w2)."""
    A = side(y[0], y[1], 1, order)
    B = side(y[2], y[3], sector, order)
    w1, w2 = y[4], y[5]
    if order == 0:
        val = []
        for a in range(3):
            for b in range(3):
                val.append(A[a][0]*B[b][0]+w1*(A[3+a][0]*B[3+b][0]-A[6+a][0]*B[6+b][0])
                           + w2*(A[3+a][0]*B[6+b][0]+A[6+a][0]*B[3+b][0]))
        return val, None, None
    pr = _prod15 if order == 2 else _prod5
    val = []
    gr = []
    he = []
    for a in range(3):
        for b in range(3):
            u1 = pr(A[a], B[b])
            u2 = _sub(pr(A[3+a], B[3+b]), pr(A[6+a], B[6+b]))
            u3 = _addl(pr(A[3+a], B[6+b]), pr(A[6+a], B[3+b]))
            val.append(u1[0]+w1*u2[0]+w2*u3[0])
            gr.append([u1[1+k]+w1*u2[1+k]+w2*u3[1+k] for k in range(4)]+[u2[0], u3[0]])
            if order == 2:
                h = [None]*21
                for (p, q), n in _S4.items():
                    h[H6[(p, q)]] = u1[5+n]+w1*u2[5+n]+w2*u3[5+n]
                for k in range(4):
                    h[H6[(k, 4)]] = u2[1+k]
                    h[H6[(k, 5)]] = u3[1+k]
                h[H6[(4, 4)]] = ZERO
                h[H6[(4, 5)]] = ZERO
                h[H6[(5, 5)]] = ZERO
                he.append(h)
    return val, gr, (he if order == 2 else None)


RMAT = arb_mat(12, 9, [arb(next((sgn for (a, b, sgn) in ray if 3*a+b == n), 0)) for ray in RAYS for n in range(9)])
MPMAT = arb_mat(3*NPL, 9, [MP[p][n][c]/4 for p in range(NPL) for c in range(3) for n in range(9)])


def _tdata(TJ, order):
    Tv, Tg, Th = TJ
    ent = []
    for n in range(9):
        ent.append(Tv[n])
        if order >= 1:
            ent += Tg[n]
        if order == 2:
            ent += Th[n]
    return arb_mat(9, (1, 7, 28)[order], ent)


def delta_part(TJ, order):
    """t_ji (12), their gradients (12x6) and Hessians (12x21, order 2) w.r.t. y, and lambda_ji = (1+t_ji)/4."""
    TT = RMAT*_tdata(TJ, order)
    t = [TT[i, 0] for i in range(12)]
    tg = [[TT[i, 1+k] for k in range(6)] for i in range(12)] if order >= 1 else None
    th = [[TT[i, 7+n] for n in range(21)] for i in range(12)] if order == 2 else None
    lam = [(1+x)/4 for x in t]
    return t, tg, th, lam


def ent_hi(lam):
    """upper bound of -x log2 x over the ball lam (true values in [0,1]; x<0 is clamped to 0)."""
    hi = lam.upper()
    if hi <= 0:
        return ZERO
    if hi < arb(0.36):                       # -t log t increases on [0,1/e], 1/e > 0.36
        return -hi*hi.log()/LN2
    return arb(0.531)                        # max_t -t log2 t = 1/(e ln 2) = 0.5307..


def unsafe_set(lam):
    return [i for i in range(12) if not lam[i].lower() > 0]


def delta_eval(dp, order, unsafe=(), lam_box=None):
    """delta restricted to the rays not in `unsafe` (their entropy is bounded naturally over the BOX enclosure `lam_box` of
    the probabilities and subtracted as `pen`): returns (value, gradient, Hessian) of  EMAX - E_safe - pen  so that
    delta >= value on the box.  `dp` is the jet of the point (or box) at which the safe terms are expanded."""
    t, tg, th, lam = dp
    if unsafe and lam_box is None:
        lam_box = lam
    safe = [i for i in range(12) if i not in unsafe]
    for i in safe:
        if not lam[i].lower() > 0:
            return None
    ln = {i: lam[i].log() for i in safe}
    pen = sum((ent_hi(lam_box[i]) for i in unsafe), ZERO)/3
    val = EMAX+sum((lam[i]*ln[i] for i in safe), ZERO)/(3*LN2)-pen
    if order == 0:
        return val, None, None
    D = {i: (1+ln[i])/(12*LN2) for i in safe}
    ns = len(safe)
    TG = arb_mat(ns, 6, [tg[i][k] for i in safe for k in range(6)])
    Drow = arb_mat(1, ns, [D[i] for i in safe])
    gm = Drow*TG
    g = [gm[0, k] for k in range(6)]
    if order == 1:
        return val, g, None
    WTG = arb_mat(ns, 6, [tg[i][k]/(48*LN2*lam[i]) for i in safe for k in range(6)])
    H1 = TG.transpose()*WTG
    TH = arb_mat(ns, 21, [th[i][n] for i in safe for n in range(21)])
    H2 = Drow*TH
    h = [H1[a, b]+H2[0, n] for n, (a, b) in enumerate(PAIRS6)]
    return val, g, h


def plane_mats(TJ, order):
    """m_P (value, gradient, Hessian) for all planes at once: matrix (3*NPL) x cols."""
    return MPMAT*_tdata(TJ, order)


def plane_eval(p, TJ, dl, order, MM=None):
    """G_P value, gradient, Hessian at the point/box described by TJ; dl = delta_eval output.  None if undefined."""
    if MM is None:
        MM = plane_mats(TJ, order)
    mv = [MM[3*p+c, 0]+(arb(1)/4 if c < 2 else ZERO) for c in range(3)]
    mg = [[MM[3*p+c, 1+k] for k in range(6)] for c in range(3)]
    mh = [[MM[3*p+c, 7+n] for n in range(21)] for c in range(3)] if order == 2 else None
    m11, m22, m12 = mv
    D = m11*m22-m12*m12
    if not D.lower() > 0:
        return None
    sD = D.sqrt()
    F = (m11+m22)/2+sD
    if not F.lower() > 0:
        return None
    sF = F.sqrt()
    Dk = [mg[0][k]*m22+m11*mg[1][k]-2*m12*mg[2][k] for k in range(6)]
    Fk = [(mg[0][k]+mg[1][k])/2+Dk[k]/(2*sD) for k in range(6)]
    val = dl[0]-CSTAR*(2-2*sF)
    g = [dl[1][k]+CSTAR*Fk[k]/sF for k in range(6)]
    if order == 1:
        return val, g, None
    h = [None]*21
    for n, (a, b) in enumerate(PAIRS6):
        Dkl = (mh[0][n]*m22+mg[0][a]*mg[1][b]+mg[0][b]*mg[1][a]+m11*mh[1][n]
               - 2*(mg[2][a]*mg[2][b]+m12*mh[2][n]))
        Fkl = (mh[0][n]+mh[1][n])/2+Dkl/(2*sD)-Dk[a]*Dk[b]/(4*D*sD)
        h[n] = dl[2][n]-CSTAR*(-Fkl/sF+Fk[a]*Fk[b]/(2*F*sF))
    return val, g, h


def ball(lo, hi):
    return arb(lo).union(arb(hi))
