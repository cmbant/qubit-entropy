#!/usr/bin/env python3
"""NU-1--NU-3: rational algebra for the analytic near-uniform dual bound.

Proved in: sup_renyi.tex (lem:fqe-near-uniform-moment, lem:fqe-moment-rectangle,
lem:fqe-near-uniform-dual); NU-3 checks the NU column of tab:fqe-moment-rectangle
through moment_rectangle.check_rectangle. No intervals, sampling or subdivision are used.
The Taylor remainder, moment projection, log-concavity and strong-convexity
arguments are proved in the text. Every finite numerical inequality below
is an exact rational comparison.
"""
from fractions import Fraction as F
import moment_rectangle as MR
from moment_rectangle import coefficients

QA = F(1587, 1000)
X, D, V0 = F(1, 25), F(3, 25), F(1, 20)
LLO, LHI = F(6931, 10000), F(1733, 2500)
ROWS = {
    'HT': (QA, (F(35425, 10000), F(1872, 10000), F(11235, 10000)),
           F(21, 2500), F(1, 50), F(1, 125)),
}

def log_lower(z):
    assert 0 <= z < 1
    return z-z*z/2+z**3/3-z**4/4

def prove_near_uniform(q, theta):
    """Check the printed HT row; return its proved slack per p-1/4."""
    matches = [row for row in ROWS.values() if row[0] == q and row[1] == theta]
    assert len(matches) == 1
    _, _, fmin, eta, margin = matches[0]
    al, be, ga = theta
    assert q == QA and 0 <= ga <= F(6, 5)
    A, B, C, D5 = coefficients(q)
    assert F(9, 5) <= A <= F(19, 10)
    assert -F(21, 20) <= B < 0 and F(7, 5) <= C <= F(3, 2)
    assert -F(12, 5) <= D5 < 0
    c, b = al+8*be-ga/2, 12*be
    # The NU column of tab:fqe-moment-rectangle is this row; the shared check
    # covers (MR1) but Q, (MR2), and (MR3) at v_* = V0 with k=4 and L in (LLO, LHI).
    col = MR.TABLE['NU']
    assert (col['X'], col['vmax'], -col['vmin'], col['vstar']) == (X, D, D, V0)
    assert (col['b'], col['c'], col['gamma']) == (b, c, ga)
    assert (col['f0'], col['eta'], col['k'], col['L']) == (fmin, eta, 4, (LLO, LHI))
    assert (col['Q'], col['m'], col['S'], col['kappa']) == (F(837, 500)**2-F(4, 3), F(6, 5), F(753, 500), 5)
    assert (col['Nplus'], col['Vplus'], col['Wminus']) == (F(13, 12), F(41, 20), 0)
    V = A+B*V0+C*V0*V0
    W = C/4+D5*V0
    N = 1+X*V+X*X*W
    assert log_lower(N-1) == sum((-1)**(j+1)*(N-1)**j/j for j in range(1, 5))
    mu = MR.check_rectangle('NU')
    assert mu == fmin-eta*eta/10 > margin
    return margin

def main():
    import sympy as s
    a, b, c, x, y = s.symbols('a b c x y')
    w = (a, b, c, -a-b-c)
    p2, p3, p4, p5 = [sum(v**k for v in w) for k in (2, 3, 4, 5)]
    assert s.expand(p5-s.Rational(5, 6)*p2*p3) == 0
    residual = sum((v*v-p2/4-p3*v/p2)**2 for v in w)
    assert s.factor(residual-(p4-p2*p2/4-p3*p3/p2)) == 0
    q = s.symbols('q')
    exact = [s.prod(q-j for j in range(k))*4**(k-1)/s.factorial(k) for k in (2, 3, 4, 5)]
    A, B, C, D5 = coefficients(q)
    for lhs, rhs in zip(exact, (A, B, C, D5*s.Rational(6, 5))):
        assert s.expand(lhs-rhs) == 0
    print('NU-1 PASS: fifth-order moment identities and projection remainder, exact symbolic')

    # Elementary atanh series proves the only logarithm constants in the proof.
    series = 2*sum(F(1, (2*k+1)*3**(2*k+1)) for k in range(10))
    tail = F(2, 21*3**21)*F(9, 8)
    assert LLO < series < series+tail < LHI < F(7, 10)
    a0, b0, c0, t0 = coefficients(QA)
    assert F(9, 5) < a0 < F(19, 10)
    assert b0 < 0 and -b0 < F(21, 20) and F(7, 5) < c0 < F(3, 2)
    assert -t0 < F(12, 5)
    assert F(1, 75) < D*D
    assert F(9, 5)-F(21, 20)*D == F(837, 500)
    assert F(19, 10)+F(21, 20)*D+F(3, 2)*D*D < F(41, 20)
    assert F(7, 20)-F(12, 5)*D > 0
    assert F(3, 8)+F(12, 5)*D < F(2, 3)
    assert 1+X*F(41, 20)+X*X*F(2, 3) < F(13, 12)
    assert F(837, 500)**2-F(4, 3) > F(6, 5)*F(13, 12)**2
    assert MR.RHO == F(3, 5)*F(7, 10) and F(587, 1000) < F(3, 5) and LHI < F(7, 10)
    assert 2*MR.RHO*F(6, 5) < F(6, 5)                  # m = 6/5 > 2 rho gamma for 0 <= gamma <= 6/5
    slope = F(21, 20)+3*D+X*F(12, 5)
    assert slope == F(753, 500)
    assert (F(14, 5)-X*slope*slope)/(MR.RHO*F(13, 12)**2) > 5
    print('NU-2 PASS: rational log bounds and the (MR1)-(MR2) constants of the NU column; no subdivisions')

    import verify_renyi_cluster_sliver as HT
    assert ROWS['HT'][:2] == (HT.QA, HT.THETA_A)
    for name, (q, theta, _, _, _) in ROWS.items():
        margin = prove_near_uniform(q, theta)
        print(f'NU-3 PASS: {name} moment-rectangle column; analytic dual slack >= ({margin})*(p-1/4)')

if __name__ == '__main__':
    main()
