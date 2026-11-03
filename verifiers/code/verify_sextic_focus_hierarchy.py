#!/usr/bin/env python3
"""SF-1/SF-5: exact spin-flip sextic covariant and icosahedral Gram checks.

Integral/Fraction arithmetic only, no numerical solver or floating point.
At a contracted site L/M and spin flip give a negative Kronecker delta.
For a focused site, Sym^3 identifies replica triples by Dicke weight,
whose squared norm is weighted by 1/binom(3, weight).
"""
from collections import defaultdict
from itertools import combinations, product
from fractions import Fraction as Q
from math import comb

PENT = {(1, 3), (2, 3), (2, 4), (4, 5), (1, 5)}

def ico_support():
    a = []
    for pair in combinations(range(6), 2):
        sg = -1 if pair in PENT else 1
        x, y = [0]*7, [1]*7
        for j in pair:
            x[j+1] = 1
            y[j+1] = 0
        a.extend(((tuple(x), 0, sg), (tuple(y), 0, -sg)))
    a += [((0, 1, 1, 1, 1, 1, 1), 1, 1),
          ((1, 0, 0, 0, 0, 0, 0), 1, 1)]
    return a

def contractions(amplitudes, n, I):
    """Integer coefficients indexed by (r, beta degree, Dicke output)."""
    I = tuple(sorted(I))
    R = tuple(j for j in range(n) if j not in I)
    out = defaultdict(int)
    for x, tx, sx in amplitudes:
        for y, ty, sy in amplitudes:
            for u, tu, su in amplitudes:
                allowed = []
                for j in R:
                    opts = []
                    if y[j] == u[j]:
                        opts.append((0, x[j]))  # L
                    if x[j] == u[j]:
                        opts.append((1, y[j]))  # M
                    if not opts:
                        break
                    allowed.append(opts)
                else:
                    weights = tuple(x[j]+y[j]+1-u[j] for j in I)
                    phase = (-1)**(len(R)+sum(u[j] == 0 for j in I))*sx*sy*su
                    for opts in product(*allowed):
                        r = sum(v[0] for v in opts)
                        bits = tuple(v[1] for v in opts)
                        out[(r, tx+ty+tu, (weights, bits))] += phase
    return {key: val for key, val in out.items() if val}

def gram_poly(amplitudes, n, I):
    D = contractions(amplitudes, n, I)
    f = len(I)
    k = (n-f)//2+1
    outputs = {v for r, b, v in D}
    C = [[[0]*7 for _ in range(k)] for _ in range(k)]
    for v in outputs:
        den = 1
        for w in v[0]:
            den *= comb(3, w)
        assert (3**f) % den == 0
        mult = (3**f)//den
        for i in range(k):
            for j in range(k):
                for p in range(4):
                    a = D.get((i, p, v), 0)
                    if not a:
                        continue
                    for q in range(4):
                        C[i][j][p+q] += mult*a*D.get((j, q, v), 0)
    return C, 3**f

def ico_gram(I):
    C, den = gram_poly(ico_support(), 7, I)
    k = len(C)
    G = []
    for i in range(k):
        row = []
        for j in range(k):
            co = C[i][j]
            assert co[1] == co[3] == co[5] == 0, (I, i, j, co)
            num = sum(co[d]*15**(3-d//2)*37**(d//2) for d in (0, 2, 4, 6))
            row.append(Q(num, den*524**3))
        G.append(row)
    return G

def real_purity_moments(amp, n, S):
    R = tuple(j for j in range(n) if j not in S)
    blocks = defaultdict(list)
    for bits, value in amp.items():
        rest = tuple(bits[j] for j in R)
        local = sum(bits[j] << i for i, j in enumerate(S))
        blocks[rest].append((local, value))
    m = 2**len(S)
    rho = [[0]*m for _ in range(m)]
    for block in blocks.values():
        for i, vi in block:
            for j, vj in block:
                rho[i][j] += vi*vj
    purity = sum(rho[i][j]*rho[j][i]
                 for i in range(m) for j in range(m))
    cubic = sum(rho[i][j]*rho[j][k]*rho[k][i]
                for i in range(m) for j in range(m) for k in range(m))
    return purity, cubic

def four_qubit_regression():
    n = 4
    ints = [2, -1, 0, 1, 2, 1, -1, 0, 1, 0, 2, -2, 1, -1, 1, 2]
    amp = {tuple((idx >> j) & 1 for j in range(n)): v
           for idx, v in enumerate(ints) if v}
    N = sum(v*v for v in amp.values())
    P = T = 0
    for j in (1, 2, 3):
        p, t = real_purity_moments(amp, n, (0, j))
        P += p
        T += t
    F = 17*N*P - 12*T - 12*N**3
    G00 = G01 = G11 = Q(0)
    for i in range(n):
        C, den = gram_poly([(bits, 0, v) for bits, v in amp.items()],
                           n, (i,))
        G00 += Q(C[0][0][0], den)
        G01 += Q(C[0][1][0], den)
        G11 += Q(C[1][1][0], den)
    assert 32*F == 45*G00+3*(225*G00-120*G01+16*G11)
    print('SF-1 PASS: exact 4-qubit sextic with residual 4 Phi_1 - 15 Phi_0')

def zero_focus_stationarity():
    """The empty-focus Gram gives the family-wide purity-gradient Wronskian and stationarity at Psi_ico."""
    import sympy as s
    C, den = gram_poly(ico_support(), 7, ())
    G = ico_gram(())
    assert len(G) == 4 and G[0][0] == 1
    for j in range(4):
        assert G[1][j] == Q(7, 2)*G[0][j]
        assert G[3][j] == Q(605, 131)*G[0][j]
    assert G[0][0]*G[2][2]-G[0][2]**2 == Q(74925, 294499921)
    x = s.symbols('x')
    u = (1-2*x)/30
    q = s.Matrix([1, s.Rational(7, 2), (832*x*x-112*x+163)/30,
                  (2096*x*x-296*x+149)/30])
    dq = q.diff(x)
    closed = (q*q.T+x*(1-2*x)*dq*dq.T/8).applyfunc(s.expand)
    tensor = s.Matrix(4, 4, lambda i, j: sum(
        s.Rational(C[i][j][d], den)*u**(3-d//2)*x**(d//2)
        for d in (0, 2, 4, 6)
    )).applyfunc(s.expand)
    assert closed == tensor
    assert max(s.degree(value, x) for value in closed) <= 3
    assert s.expand(131*dq[2]-52*dq[3]) == 24
    frame = s.Matrix([[1, 0], [0, 0], [0, 131], [0, -52]])
    assert s.expand((frame.T*closed*frame).det()-72*x*(1-2*x)) == 0
    for w in [s.Matrix([-7, 2, 0, 0]), s.Matrix([
        96*x*x+3628*x-263, 0, -2*(524*x-37), 4*(104*x-7)
    ])]:
        assert (closed*w).applyfunc(s.expand) == s.zeros(4, 1)
    for endpoint in [0, s.Rational(1, 2)]:
        assert closed.subs(x, endpoint).rank() == 1
    # One-site focus Phi_{i,0}(psi_x) = 0 identically for both site orbits:
    for site in (0, 1):
        Cs, _ = gram_poly(ico_support(), 7, (site,))
        assert Cs[0][0] == [0]*7
    print('SF-5 PASS: every coefficient of the tensor Gram equals '
          'q q^T + x(1-2x) qprime qprime^T/8; no determining-point interpolation')
    print('SF-5 PASS: fixed two-source Gram determinant 72x(1-2x), '
          'rank2 on the open family and rank1 at both endpoints; '
          'both kernel rows and purity-gradient value605/131 are exact')

def scalar_basis_controls():
    """Exact character and Krawtchouk controls for the analytic all-n basis."""
    import sympy as s
    for n in range(1, 8):
        even = [a for a in range(2**n) if a.bit_count() % 2 == 0]
        representatives = range(2**(n-1))
        characters = s.Matrix([
            [(-1)**((a & subset).bit_count()) for a in even]
            for subset in representatives
        ])
        assert characters*characters.T == 2**(n-1)*s.eye(2**(n-1))
    for n in range(1, 18):
        m = n//2
        krawtchouk = s.Matrix([
            [sum((-1)**j*comb(k, j)*comb(n-k, r-j)
                 for j in range(max(0, r-(n-k)), min(k, r)+1))
             for r in range(m+1)]
            for k in range(0, n+1, 2)
        ])
        assert krawtchouk.rows == m+1 and krawtchouk.det() != 0
    print('SF-5 PASS: even-subset Fourier bases through n7 and symmetric '
          'Krawtchouk bases through n17; all-n scalar-gradient completeness '
          'is proved analytically in prop:zero-focus')

def main():
    four_qubit_regression()
    scalar_basis_controls()
    zero_focus_stationarity()

if __name__ == '__main__':
    main()
