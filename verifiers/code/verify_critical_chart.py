#!/usr/bin/env python3
"""TC: the correlation-matrix chart of the critical locus (critical_sharp_coefficient.tex, Section 'Critical chart').

TC-1  exact (sympy): the twelve cut rays M h_ji are Bell rays with no local terms and signed-permutation correlation
      matrices C^{ji}, equal to the table used by critical_cells.py; lambda_ji = (1+<C^{ji},T>)/4 exactly for a
      random unnormalised rational v (T the correlation matrix of Mv).
TC-2  exact: A = Re(v v^+)/|v|^2 equals 1/4 I + 1/4 sum_ji t_ji B_ji;  F_P = 1/2 tr m_P + sqrt det m_P (m_P = V^T A V)
      is the larger overlap of the two HS rays of the plane P (numerical, 1e-12, 200 random states).
TC-3  numerical regression: T has singular values (1,k,k), det T = -k^2, k^2 = tau = |v^T v|^2.
TC-4  chart completeness (numerical regression of Lemma 'chart'): 4000 random critical states, reduced by the symmetry
      group (cyclic shifts, even sign flips) to (P,Q,P',Q',w1,w2) in the chart; T and G are reproduced; the transposition
      acts as (P,Q,P',Q',w1,w2) -> (P',Q',P,Q,w1,sg w2).
TC-5  float twin and ball arithmetic agree (values; gradients and Hessians against finite differences).
TC-6  analytic range tau >= 0.36: the entropy and fidelity bounds already imply delta >= C* d^2 (the reduced function of u is
      concave, analytic; its two endpoint values are enclosed in ball arithmetic).
"""
import itertools
import numpy as np
import sympy as sp
from fractions import Fraction
from flint import arb
import critical_cells as cc
import critical_float as cf

rng = np.random.default_rng(20)
M = sp.Matrix([[1, 0, 0, sp.I], [0, sp.I, 1, 0], [0, sp.I, -1, 0], [1, 0, 0, -sp.I]])/sp.sqrt(2)
PA = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]), sp.diag(1, -1)]
SP = [[sp.kronecker_product(p, q) for q in PA] for p in PA]
HH = [sp.eye(4), sp.Matrix(cc._H1)/2, sp.Matrix(cc._H2M)/2]

# ---- TC-1 -------------------------------------------------------------------------------------------------------
table = {}
for j in range(3):
    for i in range(4):
        b = M*HH[j][i, :].T
        Pr = b*b.H
        assert all(sp.simplify((Pr*sp.kronecker_product(p, sp.eye(2))).trace()) == 0 for p in PA)
        assert all(sp.simplify((Pr*sp.kronecker_product(sp.eye(2), p)).trace()) == 0 for p in PA)
        C = sp.Matrix(3, 3, lambda a, c: sp.simplify((Pr*SP[a][c]).trace()))
        ent = sorted((a, c, int(C[a, c])) for a in range(3) for c in range(3) if C[a, c] != 0)
        assert ent == sorted(cc._C[(j, i)]) and len(ent) == 3, (j, i, C)
        table[(j, i)] = C
v = sp.Matrix([sp.Rational(int(x), 7) + sp.I*sp.Rational(int(y), 5) for x, y in rng.integers(-9, 10, (4, 2))])
psi = M*v
Tt = sp.Matrix(3, 3, lambda a, c: sp.simplify((psi.H*SP[a][c]*psi)[0]))
nrm = sp.simplify((psi.H*psi)[0])
for (j, i), C in table.items():
    lam = sp.simplify(abs((HH[j][i, :]*v)[0])**2)
    assert sp.simplify(lam-(nrm+sum(C[a, c]*Tt[a, c] for a in range(3) for c in range(3)))/4) == 0
print('[PASS] TC-1 the twelve Bell rays M h_ji have no local terms and signed-permutation correlation matrices '
      '(table of critical_cells.py); lambda_ji = (|v|^2 + <C^{ji},T>)/4 exactly')

# ---- TC-2 / TC-3 ------------------------------------------------------------------------------------------------
Hn = [np.array(h.tolist(), float) for h in HH]
Mn = np.array(M.evalf(30).tolist(), complex)
pn = [np.array(p.tolist(), complex) for p in PA]
Sn = [[np.kron(p, q) for q in pn] for p in pn]


def T_of(v):
    ps = Mn@v
    return np.array([[np.real(ps.conj()@Sn[a][c]@ps) for c in range(3)] for a in range(3)])


def t_of(T):
    return np.array([sum(s*T[a, b] for a, b, s in ray) for ray in cc.RAYS])


Bn = np.array([np.outer(h, h) for j in range(3) for h in Hn[j]])
hv = np.array([[float(x) for x in r] for r in cc.RAYROWS])
etas = []
for i in range(4):
    for eps in itertools.product((1, -1), repeat=3):
        e = np.zeros(4, complex)
        e[i] = 1/np.sqrt(2)
        for n_, j in enumerate([k for k in range(4) if k != i]):
            e[j] = 1j*eps[n_]/np.sqrt(6)
        etas.append(e)


def planes_F(A):
    out = []
    for (k, oth, sg) in cc.PLANES:
        V = np.zeros((4, 2))
        V[k, 0] = 1
        for n_, j in enumerate(oth):
            V[j, 1] = sg[n_]/np.sqrt(3)
        m = V.T@A@V
        out.append(0.5*np.trace(m)+np.sqrt(max(np.linalg.det(m), 0)))
    return np.array(out)


worst = 0
for _ in range(200):
    v = rng.normal(size=4)+1j*rng.normal(size=4)
    v /= np.linalg.norm(v)
    A = np.real(np.outer(v, v.conj()))
    T = T_of(v)
    t = t_of(T)
    A2 = np.eye(4)/4+np.einsum('n,nab->ab', t, Bn)/4
    worst = max(worst, np.abs(A-A2).max())
    sv = np.linalg.svd(T)[1]
    kap = abs(v@v)
    assert np.allclose(sv, [1, kap, kap], atol=1e-11) and abs(np.linalg.det(T)+kap**2) < 1e-11
    q = np.array([abs(e.conj()@v)**2 for e in etas])
    assert abs(planes_F(A).max()-q.max()) < 1e-12
assert worst < 1e-12
print('[PASS] TC-2 A = 1/4 I + 1/4 sum t_ji B_ji (error < 1e-12); max_P F_P = max over the 32 HS rays (200 random states)')
print('[PASS] TC-3 T has singular values (1,k,k), det T = -k^2, k^2 = tau (200 random critical states)')

# ---- TC-4 chart completeness --------------------------------------------------------------------------------------
GEN_E = [np.eye(3)]
perm3 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)


def reduce_T(T):
    """return (y, sector, T') with T' = Sa T Sb^T in the chart, using cyclic shifts, even sign flips, transposition."""
    U, S, Vt = np.linalg.svd(T)
    kap = S[1]
    r = U[:, 0]
    s = Vt[0]
    if r@T@s < 0:
        s = -s
    Sa = np.eye(3)
    Sb = np.eye(3)
    ra, sb = r.copy(), s.copy()
    for _ in range(3):
        if abs(ra[0]) >= abs(ra[1])-1e-15 and abs(ra[0]) >= abs(ra[2])-1e-15:
            break
        Sa = perm3@Sa
        ra = perm3@ra
    for _ in range(3):
        if abs(sb[0]) >= abs(sb[1])-1e-15 and abs(sb[0]) >= abs(sb[2])-1e-15:
            break
        Sb = perm3@Sb
        sb = perm3@sb
    neg = [i for i in range(3) if ra[i] < 0]+[3+i for i in (1, 2) if sb[i] < 0]
    flips = list(neg)
    if sb[0] < 0:
        # s_1 < 0 is kept as the sector unless the number of flips is odd
        pass
    cnt = len(flips)
    sgn1 = np.sign(sb[0]) if sb[0] != 0 else 1.0
    if cnt % 2 == 1:
        flips.append(3)
        sgn1 = -sgn1
    Da = np.ones(3)
    Db = np.ones(3)
    for f in flips:
        if f < 3:
            Da[f] = -1
        else:
            Db[f-3] = -1
    Sa = np.diag(Da)@Sa
    Sb = np.diag(Db)@Sb
    Tn = Sa@T@Sb.T
    return Tn, int(sgn1)


def chart_T(y, sector):
    return cf.T_f(np.array(y, float), sector)


worst_T = worst_G = 0
for it in range(4000):
    v = rng.normal(size=4)+1j*rng.normal(size=4)
    if it % 7 == 0:
        v[rng.integers(0, 4)] = 0
    v /= np.linalg.norm(v)
    T = T_of(v)
    Tn, sector = reduce_T(T)
    U, S, Vt = np.linalg.svd(Tn)
    r = U[:, 0]
    s = Vt[0]
    if r@Tn@s < 0:
        s = -s
    if r[0] < 0:
        r, s = -r, -s
    P, Q = r[1]/r[0], r[2]/r[0]
    Pp, Qp = s[1]/abs(s[0]), s[2]/abs(s[0])
    assert -1e-9 <= P <= 1+1e-9 and -1e-9 <= Q <= 1+1e-9 and -1e-9 <= Pp <= 1+1e-9 and -1e-9 <= Qp <= 1+1e-9, (r, s)
    y0 = np.array([P, Q, Pp, Qp, 0, 0.0])
    # sector sign = sign of s_1 after normalising r_1>0 (flip both r,s if needed): chart has s = (sector,P',Q')/N'
    sector = int(np.sign(s[0])) if s[0] != 0 else 1
    ra, xa, ya = cf.side_f(np.array(P), np.array(Q), np.array(1.0))
    sb_, xb, yb = cf.side_f(np.array(Pp), np.array(Qp), np.array(float(sector)))
    Q1 = np.outer(xa, xb)-np.outer(ya, yb)
    Q2 = np.outer(xa, yb)+np.outer(ya, xb)
    Tr = Tn-np.outer(ra, sb_)
    w1, w2 = (Tr*Q1).sum()/2, (Tr*Q2).sum()/2
    worst_T = max(worst_T, np.abs(chart_T([P, Q, Pp, Qp, w1, w2], sector)-Tn).max())
    A = np.real(np.outer(v, v.conj()))
    Gc = cf.Gp_f(np.array([[P, Q, Pp, Qp, w1, w2]]), sector).max()
    d2 = 2-2*np.sqrt(planes_F(A).max())
    Eg = -sum(x*np.log2(x) for x in (np.array([Hn[j]@A@Hn[j].T for j in range(3)]).diagonal(axis1=1, axis2=2).ravel()) if x > 0)/3
    G0 = cc.EMAX.mid()
    Gtrue = float(G0)-Eg-cf.CF*d2
    worst_G = max(worst_G, abs(Gc-Gtrue))
    assert abs(w1*w1+w2*w2-abs(v@v)**2) < 1e-9
assert worst_T < 1e-9 and worst_G < 1e-9
print('[PASS] TC-4 chart completeness: 4000 random critical states reduced to the chart; T and G reproduced (error < 1e-9), '
      '|w|^2 = tau')

# transposition acts on the chart as (P,Q,P',Q',w1,w2) -> (P',Q',P,Q,w1,sg*w2)
yy = np.concatenate([rng.uniform(0, 1, (500, 4)), rng.uniform(-0.6, 0.6, (500, 2))], 1)
for sg in (1, -1):
    y2 = np.stack([yy[:, 2], yy[:, 3], yy[:, 0], yy[:, 1], yy[:, 4], sg*yy[:, 5]], 1)
    assert np.abs(cf.Gp_f(yy, sg).max(1)-cf.Gp_f(y2, sg).max(1)).max() < 1e-12
print('[PASS] TC-4 (transposition): G(P,Q,P\',Q\',w1,w2) = G(P\',Q\',P,Q,w1,sg w2) in both parity sectors (500 random points each)')

# ---- TC-5 float twin vs ball arithmetic ----------------------------------------------------------------------------
from critical_cells import T_jets, delta_part, delta_eval, plane_eval
errv = errg = errH = 0
for it in range(30):
    sector = 1 if it % 2 == 0 else -1
    y = np.concatenate([rng.uniform(0.05, 0.95, 4), rng.uniform(-0.5, 0.5, 2)])
    TJ = T_jets([arb(float(x)) for x in y], sector, 2)
    dl = delta_eval(delta_part(TJ, 2), 2, ())
    pl = [plane_eval(p, TJ, dl, 2) for p in range(16)]
    gf = cf.Gp_f(y[None], sector)[0]
    errv = max(errv, max(abs(float(pl[p][0].mid())-gf[p]) for p in range(16)))
    p = int(np.argmax(gf))
    h = 1e-5
    fd = np.array([(cf.Gp_f((y+h*np.eye(6)[k])[None], sector)[0, p]-cf.Gp_f((y-h*np.eye(6)[k])[None], sector)[0, p])/(2*h)
                   for k in range(6)])
    errg = max(errg, np.abs(fd-np.array([float(x.mid()) for x in pl[p][1]])).max())
    hh = 1e-4
    for a, b in ((0, 1), (2, 5), (4, 4), (3, 4)):
        ea, eb = np.eye(6)[a]*hh, np.eye(6)[b]*hh
        f = lambda yy: cf.Gp_f(yy[None], sector)[0, p]
        Hn_ = (f(y+ea+eb)-f(y+ea-eb)-f(y-ea+eb)+f(y-ea-eb))/(4*hh*hh)
        errH = max(errH, abs(Hn_-float(pl[p][2][cc.H6[(a, b)]].mid())))
assert errv < 1e-12 and errg < 1e-8 and errH < 1e-5
print('[PASS] TC-5 ball-arithmetic values/gradients/Hessians agree with the float twin and with finite differences '
      '(value error < 1e-12, gradient error < 1e-8, Hessian error < 1e-5; 30 random points, both sectors)')

# ---- TC-6 analytic range -------------------------------------------------------------------------------------------
# psi(u) = k(1-f) + 5k(1-u^2)/2 - 2C*(1-sqrt f), f = 1/3+u/(2 sqrt3), is concave in u (affine, -5k u^2/2 and +2C* sqrt f with
# k, C* > 0; analytic), so positivity on [0, 4/5] follows from positivity at the two endpoints, enclosed here in ball arithmetic.
k = (arb(3).log()-1)/(2*cc.LN2)
assert k > 0 and cc.CSTAR > 0
ends = []
for u in (arb(0), arb(4)/5):
    fc = arb(1)/3+u/(2*cc.S3)
    psi_ = k*(1-fc)+5*k*(1-u*u)/2-2*cc.CSTAR*(1-fc.sqrt())
    assert psi_ > 0
    ends.append(float(psi_.lower()))
print('[PASS] TC-6 for tau = 1-u^2 >= 0.36 (u <= 4/5) and F >= f_cov(tau) = 1/3+u/(2 sqrt3):  k(1-F+5 tau/2) - C*(2-2 sqrt F)'
      ' > 0 (minimum at F = f_cov); the concave function of u is positive at both endpoints: psi(0) > %.6f, psi(4/5) > %.3e'
      % tuple(ends))

# ---- TC-7 the sign symmetry of the equality point (P,Q,P',Q')=(0,0,0,1) ------------------------------------------------
zz = np.concatenate([rng.uniform(-0.2, 0.2, (2000, 3)), rng.uniform(0, 0.2, (2000, 1)), rng.uniform(-0.2, 0.2, (2000, 2))], 1)
for sg in (1, -1):
    for s1, s2 in itertools.product((1, -1), repeat=2):
        zs = zz*np.array([s1, s2, s1*s2, 1, s2, s1])
        ya_, yb_ = zz.copy(), zs.copy()
        ya_[:, 3] = 1-zz[:, 3]
        yb_[:, 3] = 1-zs[:, 3]
        assert np.abs(cf.Gp_f(ya_, sg).max(1)-cf.Gp_f(yb_, sg).max(1)).max() < 1e-12
print('[PASS] TC-7 G is invariant under z -> (s1 z1, s2 z2, s1 s2 z3, z4, s2 z5, s1 z6), z = (P,Q,P\',1-Q\',w1,w2), at the equality point '
      '(0,0,0,1) in both parity sectors (2000 random points each): the four tied planes are permuted and their gradients sum to a '
      'vector with only the z4 component, which vanishes (wall)')

# ---- TC-8 audit of the cell bound: a certified lower bound of G on a box is below G at sampled points ------------------
from critical_cover_lib import test_cell
nbox = nviol = 0
margin = 1.0
for it in range(1500):
    sg = 1 if it % 2 == 0 else -1
    wd = 2.0**(-int(rng.integers(3, 9)))
    lo_ = np.concatenate([rng.uniform(0, 1-wd, 4), rng.uniform(-0.55, 0.55-wd, 2)])
    lo_ = np.floor(lo_/wd)*wd
    lo_l, hi_l = [float(x) for x in lo_], [float(x) for x in lo_+wd]
    stt = {}
    if test_cell(lo_l, hi_l, sg, stt) != 'G':
        continue
    nbox += 1
    pts = rng.uniform(lo_l, hi_l, (200, 6))
    cor = np.array([[lo_l[k] if (m >> k) & 1 == 0 else hi_l[k] for k in range(6)] for m in range(64)])
    P_ = np.concatenate([pts, cor])
    P_ = P_[(P_[:, 4]**2+P_[:, 5]**2) < 1]
    margin = min(margin, cf.Gp_f(P_, sg).max(1).min()-stt['lb'])
    nviol += cf.Gp_f(P_, sg).max(1).min() < stt['lb']
assert nviol == 0 and nbox > 500
print('[PASS] TC-8 audit (numerical regression, not a proof): on %d random boxes certified by test_cell, the certified lower bound of G is below '
      'G at 264 sampled points and corners of each box (no violation)' % nbox)
nhs = 0
for it in range(400):
    wd = 2.0**(-int(rng.integers(9, 13)))
    lo_ = np.concatenate([1-wd*rng.integers(1, 3, 4), -wd*rng.uniform(0, 1, 2)])      # boxes containing the orbit point (1,1,1,1,0,0)
    lo_ = np.floor(lo_/(wd/2))*(wd/2)
    lo_l, hi_l = [float(x) for x in lo_], [float(x) for x in lo_+wd]
    hi_l[:4] = [min(h_, 1.0) for h_ in hi_l[:4]]
    if min(hi_l[:4]) <= max(lo_l[:4]) or test_cell(lo_l, hi_l, -1) != 'HS':
        continue
    nhs += 1
    pts = rng.uniform(lo_l, hi_l, (200, 6))
    d2s = 2-2*np.sqrt(cf.Gp_f(pts, -1, True)[2].max(1))
    assert d2s.max() <= 1/400
assert nhs > 5
print('[PASS] TC-8 audit of the HS rule: on %d random boxes near the orbit point of the sector sg=-1 certified by d^2 <= 1/400, d^2 <= 1/400 at '
      '200 sampled points of each' % nhs)


def psi_tau(tau):
    """the TC-6 function at F = f_cov(tau), for a rational tau (ball arithmetic)."""
    u = (1-arb(tau.numerator)/tau.denominator).sqrt()
    fc = arb(1)/3+u/(2*cc.S3)
    return k*(1-fc)+5*k*(arb(tau.numerator)/tau.denominator)/2-2*cc.CSTAR*(1-fc.sqrt())


assert psi_tau(Fraction(358785, 1000000)).upper() < 0 and psi_tau(Fraction(358787, 1000000)).lower() > 0
print('[PASS] TC-6 crossing: the TC-6 function is negative at tau = 0.358785 and positive at tau = 0.358787 (so the analytic bound alone '
      'reaches down to tau = 0.358786 at best; it is used only for tau >= 0.36; the covering handles |w|^2 <= 37/100)')

