"""Shared check of the hypotheses (MR1)--(MR3) of lem:fqe-moment-rectangle (sup_renyi.tex).

Used by verify_renyi_near_uniform.py (NU-3), verify_renyi_small_head.py (SHD-4)
and verify_renyi_capped_cubic.py (CG-1); one call per column of
tab:fqe-moment-rectangle (TABLE below mirrors it). Every comparison is exact rational arithmetic.

The bound Q <= V^2-2W on the rectangle is derived separately by each application
and checked in its own verifier; the other bounds in (MR1) are checked here from
the exact coefficients (V convex and W, B+2Cv+XD5 linear in v, so their extremes
are at the ends of the v-interval or at the vertex of V).
"""
from fractions import Fraction as F

QS = F(1587, 1000)
RHO = F(21, 50)          # (3/5)(7/10) > rL


def coefficients(q=QS):
    r = q-1
    A = 2*q*r
    B = F(8, 3)*q*r*(q-2)
    C = F(8, 3)*q*r*(q-2)*(q-3)
    D5 = F(2, 3)*(q-4)*C
    return A, B, C, D5


# The columns of tab:fqe-moment-rectangle. HT is the cluster-sliver multiplier
# row (alpha, beta, gamma) of tab:fqe-dual-co; SHD uses (397/100, 3/20, 159/100).
HT = (F(35425, 10000), F(1872, 10000), F(11235, 10000))
LOG4 = (F(6931, 10000), F(1733, 2500))
LOG8 = (F(693147, 10**6), F(693148, 10**6))
TABLE = {
    'NU': dict(X=F(1, 25), vmin=-F(3, 25), vmax=F(3, 25), vstar=F(1, 20),
               b=12*HT[1], c=HT[0]+8*HT[1]-HT[2]/2, gamma=HT[2],
               Nplus=F(13, 12), Vplus=F(41, 20), Wminus=F(0),
               Q=F(837, 500)**2-F(4, 3), m=F(6, 5), S=F(753, 500), kappa=F(5),
               f0=F(21, 2500), eta=F(1, 50), k=4, L=LOG4),
    'SHD': dict(X=F(3, 32), vmin=F(0), vmax=F(9, 50), vstar=F(27, 200),
                b=F(9, 5), c=F(35, 8), gamma=F(159, 100),
                Nplus=F(6, 5), Vplus=F(19, 10), Wminus=F(3, 50),
                Q=F(8, 3), m=F(7, 4), S=F(19, 10), kappa=F(4),
                f0=F(1, 800), eta=F(1, 100), k=8, L=LOG8),
    'CG': dict(X=F(17, 100), vmin=-F(6, 25), vmax=F(6, 25), vstar=F(0),
               b=F(8, 3), c=F(171, 40), gamma=F(11, 10),
               Nplus=F(3, 2), Vplus=F(9, 4), Wminus=F(21, 100),
               Q=F(8, 3), m=F(1), S=F(213, 100), kappa=F(2),
               f0=F(1, 250), eta=F(1, 20), k=8, L=LOG8),
}


def check_rectangle(name):
    """Check (MR1) except the Q bound, (MR2) and (MR3) for one table column.

    Returns mu = f0 - eta^2/(2 kappa) as an exact rational.
    """
    col = TABLE[name]
    log_terms, (L_lo, L_hi) = col['k'], col['L']
    q = QS
    r = q-1
    A, B, C, D5 = coefficients(q)
    X, lo, hi, vs = col['X'], col['vmin'], col['vmax'], col['vstar']
    b, c, ga = col['b'], col['c'], col['gamma']
    assert X > 0 and lo <= vs <= hi and ga >= 0 and C > 0
    assert log_terms % 2 == 0 and 0 < L_lo < L_hi and r*L_hi < RHO
    V = lambda v: A+B*v+C*v*v
    W = lambda v: C/4+D5*v
    # (R1): V is convex in v, W and B+2Cv+XD5 are linear.
    vertex = min(max(-B/(2*C), lo), hi)
    Vmin, Vmax = V(vertex), max(V(lo), V(hi))
    Wmin, Wmax = min(W(lo), W(hi)), max(W(lo), W(hi))
    assert Vmax <= col['Vplus'] and Wmin >= -col['Wminus'] and col['Wminus'] >= 0
    assert Vmin-X*max(-Wmin, 0) >= 0                       # V+xW >= 0, so N >= 1
    assert 1+X*Vmax+X*X*max(Wmax, 0) <= col['Nplus']
    assert max(abs(B+2*C*lo+X*D5), abs(B+2*C*hi+X*D5)) <= col['S']
    # (R2)
    Np, m, S, ka = col['Nplus'], col['m'], col['S'], col['kappa']
    assert col['Q']-2*X*col['Vplus']*col['Wminus'] >= m*Np*Np
    assert m > 2*RHO*ga
    assert ka > 0 and 2*C-X*S*S >= ka*RHO*Np*Np
    # (R3) at v_*: alternating lower sum for ln(1+z), L between L_lo and L_hi.
    z = X*V(vs)+X*X*W(vs)
    assert 0 < z < 1
    lower = sum((-1)**(j+1)*z**j/j for j in range(1, log_terms+1))
    assert lower > 0
    assert lower/(r*L_hi*X)+b*vs-c+ga*X > col['f0']
    for L in (L_lo, L_hi):                                 # Phi_v is monotone in L
        assert abs((B+2*C*vs+X*D5)/(r*L*(1+z))+b) < col['eta']
    return col['f0']-col['eta']**2/(2*ka)
