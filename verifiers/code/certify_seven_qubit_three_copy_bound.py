#!/usr/bin/env python3
"""TCB-0--TCB-4: exact primal-dual certificate that the seven-qubit three-copy
spin-flip relaxation has exact optimum L_3 = 125/978 > 639/5000 = 0.1278 (> 1/8,
the AME value), so every pure seven-qubit state has mean three-body purity
at least 125/978.

Statement: thm:threecopy (in replica_hierarchy.tex)
The certificate uses the exact boundary optimal pseudo-moment m*/70416 (37 four-digit
integers) and complementary slackness: on the 6 active blocks b where B_b(m*) has a
non-trivial kernel V_b = ker B_b(m*), the 25 canonically normalized S_m-invariant
projected orbit matrices Y_hat_{b,p} = M_{b,0}^{-1} K_{b,p} M_{b,0}^{-1} / Tr(...)
span the dual kernel space independently of the choice of kernel basis V_b. Evaluating
the 14 reduced dual equations at 12 simple rational free parameters yields strictly
positive definite kernel multipliers Y_b > 0 and an exact 37-orbit dual identity
  pi_3 - 125/978 = sum_{b in A} <Y_b, V_b^T B_b(mu) V_b> + nu . (antisymmetrizer relations)(mu)
with no external certificate file.

TCB-0 checks the one-site algebra; TCB-1 checks soundness of the block kernels on
genuine states (n=4: random states, exact orbit averaging over all words;
n=7: random permutation-symmetric states); TCB-2 checks primal feasibility of m*/70416
and the exact 37-orbit dual identity at lambda = 125/978; TCB-3 checks positive
definiteness of all 6 active-kernel dual multipliers Y_b (and of the 15 primal pivot
submatrices of B_b(m*)) by fraction-free Bareiss elimination (Sylvester); TCB-4 checks
the resulting bound L_3 = 125/978 > 639/5000 > 1/8.
"""
import math, os, sys, time
from fractions import Fraction as Fr
from math import comb
import flint
import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import three_copy_relaxation as tc

assert tc.check_algebra()
print('TCB-0 PASS: S_3 products, vanishing qubit antisymmetrizer, spin-flip expansions 6t(pi) = sum C6 sigma', flush=True)


def block_matrices(d, mu):
    return [np.tensordot(mu, b['K'].astype(float), axes=1) for b in d['blocks']]


def bareiss_pd(M):
    M = [row[:] for row in M]
    n = len(M)
    prev = 1
    for k in range(n):
        if M[k][k] <= 0:
            return False
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) // prev
        prev = M[k][k]
    return True


# TCB-1a: n = 4, random states, exact orbit averages over all 6^4 words
d4 = tc.build(4)
rng = np.random.default_rng(2026)
worst = 1.0
for trial in range(3):
    z = rng.normal(size=16) + 1j * rng.normal(size=16)
    z /= np.linalg.norm(z)
    psi = z.reshape([2] * 4)
    tot = np.zeros(d4['NO'])
    cnt = np.zeros(d4['NO'])
    for w, o in zip(d4['W'], d4['orb']):
        tot[o] += tc.mu_word(psi, w).real
        cnt[o] += 1
    mu = tot / cnt
    assert abs(mu[d4['o_norm']] - 1) < 1e-12 and np.abs(d4['rels'] @ mu).max() < 1e-12
    for B in block_matrices(d4, mu):
        worst = min(worst, np.linalg.eigvalsh(B).min() / max(1.0, np.abs(B).max()))
assert worst > -1e-12, worst
print('TCB-1a PASS: n=4 random states satisfy all %d relations; all %d blocks PSD (min normalized eig %.2e)'
      % (len(d4['rels']), len(d4['blocks']), worst), flush=True)

# TCB-1b: n = 7, random permutation-symmetric states (mu constant on S_7-orbits)
d7 = tc.build(7)
reps = []
for c in d7['codes']:
    w = []
    for _ in range(7):
        w.append(int(c % 6))
        c //= 6
    reps.append(w[::-1])
worst = 1.0
syms = []
for trial in range(3):
    a = rng.normal(size=8) + 1j * rng.normal(size=8)
    z = np.array([a[bin(i).count('1')] / np.sqrt(comb(7, bin(i).count('1'))) for i in range(128)])
    z /= np.linalg.norm(z)
    psi = z.reshape([2] * 7)
    mu = np.array([tc.mu_word(psi, w).real for w in reps])
    syms.append(mu)
    assert abs(mu[d7['o_norm']] - 1) < 1e-12 and np.abs(d7['rels'] @ mu).max() < 1e-10
    for B in block_matrices(d7, mu):
        worst = min(worst, np.linalg.eigvalsh(B).min() / max(1.0, np.abs(B).max()))
assert worst > -1e-12, worst
print('TCB-1b PASS: n=7 random symmetric states satisfy all %d relations; all %d blocks PSD (min normalized eig %.2e)'
      % (len(d7['rels']), len(d7['blocks']), worst), flush=True)

# TCB-2: exact boundary optimal pseudo-moment m*/70416 and complementary-slackness dual identity at lam = 125/978
M_STAR = [
    70416, 35208, 17604, 17736, 17604, 8868, 4526, 4474, 9000, 8868,
    4500, 8842, 4487, 2316, 2290, 1224, 1198, 4500, 4448, 4474,
    2277, 2238, 2316, 2290, 1185, 1172, 1107, 1185, 873, 1484,
    2264, 1146, 1185, 873, 392, 795, 1484,
]
PIVOTS_MAP = {
    ('omega', 6): (0,),
    ('omega', 7): (2,),
    ('gamma', 4): (0,),
    ('gamma', 5): (0, 1),
    ('gamma', 6): tuple(range(10)),
    ('gamma', 7): tuple(range(10)),
}
FREE_PARAMS = [
    sp.Rational(1, 72),
    sp.Rational(-3, 5),
    sp.Rational(-1, 800),
    sp.Rational(-1, 172),
    sp.Rational(1, 98),
    sp.Rational(623999),
    sp.Rational(1986215),
    sp.Rational(96816911),
    sp.Rational(57809478),
    sp.Rational(-594264),
    sp.Rational(86717284),
    sp.Rational(-6836481),
]

NO = d7['NO']
rels = d7['rels']
o_obj = tc.purity_orbit(d7, 3)
o_n = d7['o_norm']
lam = Fr(125, 978)
lam_sp = sp.Rational(125, 978)

assert M_STAR[o_n] == 70416 and Fr(M_STAR[o_obj], M_STAR[o_n]) == lam
assert all(sum(int(rels[a, o]) * M_STAR[o] for o in range(NO)) == 0 for a in range(len(rels)))

N_rels = [v * sp.ilcm(*[sp.fraction(x)[1] for x in v]) for v in sp.Matrix(rels.tolist()).nullspace()]
assert len(N_rels) == 14
B_rels = np.array([[int(v[i]) for v in N_rels] for i in range(NO)], dtype=object)

t0 = time.time()
all_cols = []
active_data = []
primal_piv_submatrices = []
for bi, b in enumerate(d7['blocks']):
    key = (b['kind'], b['m'])
    K = b['K']
    sz = b['D']
    Bmat = np.tensordot(np.array(M_STAR, dtype=object), K, axes=(0, 0))
    M_int = [[int(Bmat[i, j]) for j in range(sz)] for i in range(sz)]
    _, pivs = sp.Matrix(M_int).rref()
    primal_piv_submatrices.append([[M_int[i][j] for j in pivs] for i in pivs])
    if key not in PIVOTS_MAP:
        assert len(pivs) == sz
        continue
    M_fmpz = flint.fmpz_mat(M_int)
    K_null, nullity = M_fmpz.nullspace()
    k = int(nullity)
    assert len(pivs) + k == sz
    for j_col in range(k):
        assert all(sum(M_int[r][c] * int(K_null[c, j_col]) for c in range(sz)) == 0 for r in range(sz))
    V = flint.fmpq_mat(sz, k, [K_null[i, j] for i in range(sz) for j in range(k)])
    V_site = tc._s3_basis(b['m']) if b['kind'] == 'omega' else tc._even_basis(b['m'])
    U_site = tc.okron([tc.GRAM.astype(object)] * b['m']).dot(V_site)
    G_site = flint.fmpq_mat([[int(x) for x in row] for row in (V_site.T @ U_site).tolist()])
    M0i = (V.transpose() * G_site * V).inv()
    K_q = []
    for q in range(14):
        Kq = np.tensordot(B_rels[:, q], K, axes=(0, 0))
        Kq_f = flint.fmpq_mat([[int(x) for x in row] for row in Kq.tolist()])
        K_q.append(V.transpose() * Kq_f * V)
    Y_list = []
    for p in PIVOTS_MAP[key]:
        Yp_raw = M0i * K_q[p] * M0i
        tr_pp = sum((Yp_raw[i, j] * K_q[p][j, i] for i in range(k) for j in range(k)), flint.fmpq(0))
        assert tr_pp > 0
        Yp = Yp_raw / tr_pp
        Y_list.append((len(all_cols), Yp))
        col = []
        for q in range(14):
            tr_pq = sum((Yp[i, j] * K_q[q][j, i] for i in range(k) for j in range(k)), flint.fmpq(0))
            col.append(sp.Rational(int(tr_pq.p), int(tr_pq.q)))
        all_cols.append(col)
    active_data.append((bi, key, k, V, Y_list))

A_sp = sp.Matrix(all_cols).T
rhs_sp = sp.Matrix([sp.Rational(int(B_rels[o_obj, q])) - lam_sp * sp.Rational(int(B_rels[o_n, q])) for q in range(14)])
perm = list(range(25))[::-1]
w0_perm, params_tc = (A_sp[:, perm]).gauss_jordan_solve(rhs_sp)
f0_tc = list(params_tc)
assert len(f0_tc) == 12
inv_perm = [perm.index(i) for i in range(25)]
w0_tc = sp.Matrix([w0_perm[inv_perm[i]] for i in range(25)])
w_vec = w0_tc.subs({f0_tc[11 - i]: FREE_PARAMS[i] for i in range(12)})

Yb_int_list = []
Z_float = {b: np.zeros((d7['blocks'][b]['D'], d7['blocks'][b]['D'])) for b in range(len(d7['blocks']))}
orbit_pairing = [flint.fmpq(0)] * NO
for bi, key, k, V, Y_list in active_data:
    Yb = sum((flint.fmpq(int(w_vec[idx].p), int(w_vec[idx].q)) * Yp for idx, Yp in Y_list), flint.fmpq_mat(k, k))
    assert Yb == Yb.transpose()
    den_lcm = math.lcm(*[int(Yb[i, j].q) for i in range(k) for j in range(k)])
    Yb_int = [[int((Yb[i, j] * den_lcm).p) for j in range(k)] for i in range(k)]
    Yb_int_list.append((key, Yb_int))
    Zb = V * Yb * V.transpose()
    sz = d7['blocks'][bi]['D']
    Z_float[bi] = np.array([[float(Zb[i, j].p) / float(Zb[i, j].q) for j in range(sz)] for i in range(sz)])
    K = d7['blocks'][bi]['K']
    for o in range(NO):
        Ko = K[o]
        orbit_pairing[o] += sum((Zb[i, j] * int(Ko[i, j]) for i in range(sz) for j in range(sz) if Ko[i, j]), flint.fmpq(0))

rhs_37 = sp.Matrix([
    (1 if o == o_obj else 0) - (lam_sp if o == o_n else 0) - sp.Rational(int(orbit_pairing[o].p), int(orbit_pairing[o].q))
    for o in range(NO)
])
R_T = sp.Matrix(rels.T.tolist())
nu_vec, nu_params = R_T.gauss_jordan_solve(rhs_37)
assert len(nu_params) == 0 and R_T * nu_vec == rhs_37
print('TCB-2 PASS: exact primal m*/70416 and dual identity  pi_3 - %s = sum_b <Y_b,V_b^T B_b(mu) V_b> + nu.rel(mu)  on all %d orbit coordinates'
      % (lam, NO), flush=True)

for mu in syms:  # consistency on genuine (symmetric) states
    rhs_num = sum(float(np.sum(Z_float[b] * Bm)) for b, Bm in enumerate(block_matrices(d7, mu)))
    assert abs(mu[o_obj] - float(lam) - rhs_num) < 1e-10

# TCB-3: positive definiteness by exact fraction-free Bareiss elimination
for b, B_sub in enumerate(primal_piv_submatrices):
    assert bareiss_pd(B_sub), ('primal', d7['blocks'][b]['kind'], d7['blocks'][b]['m'])
for key, Yb_int in Yb_int_list:
    n_k = len(Yb_int)
    assert all(Yb_int[i][j] == Yb_int[j][i] for i in range(n_k) for j in range(n_k))
    assert bareiss_pd(Yb_int), ('dual', key)
print('TCB-3 PASS: all 15 primal pivot blocks PSD and all %d active-kernel dual matrices Y_b SPD (sizes %s; %.1fs)'
      % (len(Yb_int_list), [len(Y) for _, Y in Yb_int_list], time.time() - t0), flush=True)
assert lam > Fr(639, 5000) > Fr(1, 8)
print('TCB-4 PASS: exact three-copy optimum L_3 = %s = %.10f > 639/5000 = 0.1278 > 1/8' % (lam, float(lam)), flush=True)
