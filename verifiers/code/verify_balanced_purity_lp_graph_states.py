#!/usr/bin/env python3
"""LP-0--LP-2, GS-1--GS-4: the Rains linear-programming bound on the mean balanced purity for
n = 4..12 qubits (exact dual certificates), and graph states attaining it for every 4 <= n <= 12 except
n = 7; GS-4 shows no seven-qubit stabilizer state reaches 121/917.

Statement: thm:lptable (in even_qubit_purity_minima.tex)

For a pure n-qubit state let A_j = sum over Pauli strings P of weight j of <P>^2 (A_0 = 1).  Then
  mean_{|S|=s} tr rho_S^2 = a_s(A) = 2^{-s} C(n,s)^{-1} sum_j C(n-j,s-j) A_j,
purity of complementary cuts agrees (a_s = a_{n-s}), A_j >= 0, and the shadow coefficients S_j of
S(x,y) = A((x+3y)/2, (y-x)/2) are >= 0 (Rains, quant-ph/9611001 / IEEE TIT 45 (1999) 2361).
LP-0 checks these three facts numerically on random states (a regression of the formulation); LP-1
checks, in exact rational arithmetic, a dual certificate
   a_k - L*A_0 = sum_s e_s (a_s - a_{n-s}) + sum_i y_i S_i + sum_j w_j A_j,   y, w >= 0,
which proves a_k >= L for every pure state (k = floor(n/2)).  LP-2 gives an exact feasible point at n = 7
with value 1/8, so there the LP bound is exactly the AME value.  GS checks the graph states exactly
(cut purity of a graph state = 2^{-rank_GF2 Gamma[S, S^c]}).
"""
import itertools
from fractions import Fraction as Fr
from math import comb
import numpy as np, sympy as sp
from scipy.optimize import linprog

def forms(n):
    a = [[Fr(comb(n-j, s-j), 2**s*comb(n, s)) if j <= s else Fr(0) for j in range(n+1)] for s in range(n+1)]
    x, y = sp.symbols('x y'); S = [[Fr(0)]*(n+1) for _ in range(n+1)]
    for j in range(n+1):
        poly = sp.Poly(sp.expand(((x+3*y)/2)**(n-j)*((y-x)/2)**j), x, y)
        for i in range(n+1):
            c = sp.Rational(poly.coeff_monomial(x**(n-i)*y**i)); S[i][j] = Fr(int(c.p), int(c.q))
    return a, S

PAULI = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1., -1.])]
def lp0(n, seed, mixed=False):
    rng = np.random.default_rng(seed)
    if mixed:
        G = rng.normal(size=(2**n, 3))+1j*rng.normal(size=(2**n, 3)); rho = G @ G.conj().T; rho /= np.trace(rho)
    else:
        z = rng.normal(size=2**n)+1j*rng.normal(size=2**n); z /= np.linalg.norm(z); rho = np.outer(z, z.conj())
    A = np.zeros(n+1)
    for P in itertools.product(range(4), repeat=n):
        M = np.ones((1, 1))
        for p in P: M = np.kron(M, PAULI[p])
        A[sum(1 for p in P if p)] += abs(np.trace(M @ rho))**2
    a, S = forms(n); err = 0.0
    R = rho.reshape((2,)*(2*n))
    for s in range(n+1):
        pur = []
        for T in itertools.combinations(range(n), s):
            keep = list(T); out = [i for i in range(n) if i not in T]
            Rt = R.transpose(keep+out+[n+i for i in keep]+[n+i for i in out]).reshape(2**s, 2**(n-s), 2**s, 2**(n-s))
            r = np.einsum('ajbj->ab', Rt)
            pur.append(np.real(np.trace(r @ r)))
        err = max(err, abs(np.mean(pur)-sum(float(a[s][j])*A[j] for j in range(n+1))))
    shadow = [sum(float(S[i][j])*A[j] for j in range(n+1)) for i in range(n+1)]
    return err, min(shadow)

def certificate(n, k):
    a, S = forms(n)
    nv = 1+2*(n+1); Aub, bub = [], []
    for j in range(n+1):
        row = np.zeros(nv); row[0] = 1.0 if j == 0 else 0.0
        for s in range(n+1): row[1+s] = float(a[s][j]-a[n-s][j])
        for i in range(n+1): row[2+n+i] = float(S[i][j])
        Aub.append(row); bub.append(float(a[k][j]))
    c = np.zeros(nv); c[0] = -1
    r = linprog(c, A_ub=np.array(Aub), b_ub=bub, bounds=[(None, None)]*(n+2)+[(0, None)]*(n+1), method='highs')
    L = Fr(-r.fun).limit_denominator(10**4)
    act = [i for i in range(n+1) if r.x[2+n+i] > 1e-9]
    tight = [j for j in range(n+1) if abs(np.array(Aub[j]) @ r.x-bub[j]) < 1e-9]
    es = sp.symbols('e0:%d' % (n+1)); ys = sp.symbols('y0:%d' % (n+1)); ws = sp.symbols('w0:%d' % (n+1))
    R = lambda f: sp.Rational(f.numerator, f.denominator)
    eqs = [R(a[k][j])-(R(L) if j == 0 else 0)-sum(R(a[s][j]-a[n-s][j])*es[s] for s in range(n+1))
           - sum(R(S[i][j])*ys[i] for i in act)-(0 if j in tight else ws[j]) for j in range(n+1)]
    unk = list(es)+[ys[i] for i in act]+[ws[j] for j in range(n+1) if j not in tight]
    sol = list(sp.linsolve(eqs, unk))[0]
    val = {u: v.subs({f: 0 for f in v.free_symbols}) for u, v in zip(unk, sol)}
    tofr = lambda v: Fr(int(sp.Rational(v).p), int(sp.Rational(v).q))
    e = [tofr(val[es[s]]) for s in range(n+1)]
    y = [tofr(val[ys[i]]) if i in act else Fr(0) for i in range(n+1)]
    return L, e, y, a, S

def check_certificate(n, k, L, e, y, a, S):
    """Exact: w_j := coefficient of A_j in a_k - L A_0 - sum e (a_s-a_{n-s}) - sum y S must be >= 0."""
    assert all(v >= 0 for v in y)
    w = [a[k][j]-(L if j == 0 else 0)-sum(e[s]*(a[s][j]-a[n-s][j]) for s in range(n+1))
         - sum(y[i]*S[i][j] for i in range(n+1)) for j in range(n+1)]
    return all(v >= 0 for v in w)

def rank2(M):
    M = M.copy() % 2; r = 0
    for c in range(M.shape[1]):
        p = next((i for i in range(r, M.shape[0]) if M[i, c]), None)
        if p is None: continue
        M[[r, p]] = M[[p, r]]
        for i in range(M.shape[0]):
            if i != r and M[i, c]: M[i] ^= M[r]
        r += 1
    return r

def mean_cut_purity(G, k):
    n = len(G); cuts = list(itertools.combinations(range(n), k)); hist = {}
    for T in cuts:
        rk = rank2(G[np.ix_(list(T), [i for i in range(n) if i not in T])]); hist[rk] = hist.get(rk, 0)+1
    return sum(Fr(c, 2**rk) for rk, c in hist.items())/len(cuts), hist

def weight_enumerator(G):
    """Exact: stabilizer group of the graph state, generators X_i Z^{N(i)}; weight distribution."""
    n = len(G); A = [0]*(n+1)
    for sub in itertools.product((0, 1), repeat=n):
        x = np.array(sub); zz = (G.T @ x) % 2
        A[int(np.sum((x | zz) > 0))] += 1
    return A

if __name__ == '__main__':
    for n in (4, 5):
        for mixed in (False, True):
            err, smin = lp0(n, 7+n, mixed)
            print('LP-0 n=%d random %s state: |a_s - formula| < 1e-12: %s; min shadow coefficient > -1e-12: %s'
                  % (n, 'mixed' if mixed else 'pure', err < 1e-12, smin > -1e-12))
    print('LP-1 exact Rains-LP dual certificates for min mean balanced purity:')
    LPV = {}
    for n in range(4, 13):
        k = n//2; L, e, y, a, S = certificate(n, k); LPV[n] = L
        print('   n=%2d k=%d  bound %-6s  certificate exact: %s   (2^-k = %s)' % (n, k, L, check_certificate(n, k, L, e, y, a, S), Fr(1, 2**k)))
    # LP-2: at n = 7 the LP bound 1/8 is attained by an exact feasible point (the weight distribution a
    # hypothetical AME(7,2) state would have), so the two-copy level gives nothing beyond the AME value.
    n = 7; a, S = forms(n); A7 = [Fr(v) for v in (1, 0, 0, 0, 35, 42, 28, 22)]
    feas = (all(v >= 0 for v in A7) and all(sum(S[i][j]*A7[j] for j in range(n+1)) >= 0 for i in range(n+1))
            and all(sum((a[s_][j]-a[n-s_][j])*A7[j] for j in range(n+1)) == 0 for s_ in range(n+1)))
    val = sum(a[3][j]*A7[j] for j in range(n+1))
    print('LP-2 n=7: A = %s is exactly LP-feasible (A_j >= 0, shadow >= 0, a_s = a_{7-s}): %s; a_3(A) = %s = LP bound %s: %s'
          % ([int(v) for v in A7], feas, val, LPV[7], feas and val == LPV[7] == Fr(1, 8)))
    def circ(n, ds):
        G = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for d in ds: G[i, (i+d) % n] = G[(i+d) % n, i] = 1
        return G
    def edges(n, E):
        G = np.zeros((n, n), dtype=np.int64)
        for i, j in E: G[i, j] = G[j, i] = 1
        return G
    GRAPHS = {4: ('path P4 (cluster)', edges(4, [(0, 1), (1, 2), (2, 3)])), 5: ('C5(1)', circ(5, [1])), 6: ('C6(2,3)', circ(6, [2, 3])),
              8: ('8 vertices, 17 edges', edges(8, [(2, 0), (2, 1), (3, 1), (3, 2), (4, 1), (4, 3), (5, 0), (5, 1), (5, 2), (5, 4), (6, 0),
                                       (6, 1), (6, 3), (6, 4), (7, 0), (7, 1), (7, 6)])),
              9: ('C9(3,4)', circ(9, [3, 4])), 10: ('C10(1,2,4)', circ(10, [1, 2, 4])),
              11: ('11 vertices, 34 edges', edges(11, [(2, 0), (2, 1), (3, 1), (4, 0), (4, 1), (4, 2), (5, 1), (5, 3), (5, 4), (6, 0), (6, 1),
                                         (6, 3), (6, 4), (7, 0), (7, 1), (7, 6), (8, 1), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (9, 2),
                                         (9, 3), (9, 6), (9, 8), (10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 9)])),
              12: ('C12(3,5,6)', circ(12, [3, 5, 6]))}
    for n, (name, G) in GRAPHS.items():
        v, hist = mean_cut_purity(G, n//2)
        print('GS-1 n=%2d graph %-22s mean balanced purity %-6s = LP bound %s: %s; cut ranks %s'
              % (n, name, v, LPV[n], v == LPV[n], dict(sorted(hist.items()))))
    G = GRAPHS[10][1]; A = weight_enumerator(G); a, _ = forms(10)
    print('GS-2 n=10: weight enumerator %s; 3-uniform (A1=A2=A3=0): %s; a_5(A) = %s'
          % (A, A[1:4] == [0, 0, 0], sum(a[5][j]*A[j] for j in range(11))))
    n = 10; P = np.zeros((n, n), dtype=np.int64)
    for i in range(5):
        for s_, t_ in ((i, (i+1) % 5), (5+i, 5+(i+2) % 5), (i, 5+i)): P[s_, t_] = P[t_, s_] = 1
    print('GS-3 comparison, Petersen graph state (n=10): mean 5|5 purity = %s' % mean_cut_purity(P, 5)[0])
    # GS-4: analytic proof of prop:stab7 via MacWilliams identities and shadow integrality:
    # 280*pi_3 = 35 + 15*A_1 + 5*A_2 + A_3 and
    # 4*S_1 + 28*(a_0-a_7) - 140*(a_1-a_6) + 252*(a_2-a_5) - 140*(a_3-a_4) = 7*A_0 + A_1 + A_2 - A_3.
    # For any 7-qubit stabilizer state, A_w, S_w in Z_{\ge 0} and a_s = a_{7-s}:
    # - if A_1 + A_2 >= 1, 280*pi_3 >= 35 + 5 = 40 > 38;
    # - if A_1 = A_2 = 0, A_3 = 7 - 4*S_1 >= 0 with S_1 in Z_{\ge 0} gives A_3 in {3, 7},
    #   so 280*pi_3 >= 35 + 3 = 38 => pi_3 >= 19/140 > 121/917.
    a7, S7 = forms(7)
    assert [280 * a7[3][j] for j in range(8)] == [35, 15, 5, 1, 0, 0, 0, 0]
    mult7 = (28, -140, 252, -140)
    assert [4 * S7[1][j] + sum(c * (a7[s][j] - a7[7 - s][j]) for c, s in zip(mult7, range(4))) for j in range(8)] == [7, 1, 1, -1, 0, 0, 0, 0]
    G7 = edges(7, [(0, 1), (0, 2), (0, 3), (0, 5), (1, 4), (1, 5), (1, 6), (2, 3), (2, 6), (3, 4)])
    A7_wit = weight_enumerator(G7)
    best, hist7 = mean_cut_purity(G7, 3)
    assert A7_wit == [1, 0, 0, 3, 29, 42, 34, 19] and best == Fr(19, 140) and hist7 == {2: 3, 3: 32}
    print('GS-4 n=7: MacWilliams/shadow-integrality minimum of pi_3 over all stabilizer states = %s (witness A=%s, cut ranks %s); 121/917 < %s: %s'
          % (best, A7_wit, dict(sorted(hist7.items())), best, Fr(121, 917) < best))
