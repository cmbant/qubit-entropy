#!/usr/bin/env python3
"""SW-0--SW-4: exact three-copy (Schur-Weyl) lower bounds on mean k-body purities of pure n-qubit states.

Statement: thm:threecopy-sw (in three_copy_schur_weyl.tex)
Data: data/three_copy_sw_certificates.json (key -> lambda, rational centre, dyadic centred multipliers Y'_b;
      the gamma multipliers are Z_b = L^{-T} Y'_b L^{-1} with L D L^T the gamma block at the centre).
SW-0 per-site algebra: S_3 homomorphism, Fourier inversion, flip involution, vanishing antisymmetriser.
SW-1 regression: the structure's exact normalisation/purity functionals at the Haar moment equal the
     Haar averages computed independently from (C^2)^{(x)3n}-free formulas (n = 4..7), and the exact
     gamma map at n = 4 is independent of the sample points (two seeds).
SW-2 for each key in the data: exact gamma map (rational), multipliers Z_b positive definite (exact LDL).
SW-3 exact residual d = oo - lam*on - Gamma^*(Z); every omega-block of d positive definite.
SW-4 conclusion: pi_k >= lam for every pure n-qubit state (data key 'n' for k = floor(n/2), 'n:k' otherwise);
     if lam > 2^-k, no pure state is k-uniform, i.e. no ((n,1,k+1))_2 code exists.
Usage: certify_three_copy_schur_weyl_bounds.py [key1,key2,...]   (default: every key in the data file)
"""
import json, math, os, sys, time
from fractions import Fraction as Fr
import flint
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import three_copy_schur_weyl as sw
from three_copy_schur_weyl import Q

def haar_value(n, k):
    """mean k-body purity averaged over Haar-random pure n-qubit states: (2^k+2^{n-k})/(2^n+1)."""
    return Fr(2**k+2**(n-k), 2**n+1)

def haar_params(st):
    """Haar moment: each omega block is the identity on the invariant subspace, M = (W Gram W^T)^{-1}."""
    x = [Q(0)]*st.NP
    for bi, (a, nu, W, WT, co) in enumerate(st.blocks['om']):
        MH = (W*st.gram(nu)*WT).inv()
        for p, (b, k, l) in enumerate(st.params):
            if b == bi: x[p] = MH[k, l]
    return x

def check_sw1():
    for n in (4, 5, 6, 7):
        st = sw.Structure(n); on, oo = st.functionals(n//2); x = haar_params(st)
        N = sum((on[p]*x[p] for p in range(st.NP)), Q(0)); O = sum((oo[p]*x[p] for p in range(st.NP)), Q(0))
        h = haar_value(n, n//2)
        assert O/N == Q(h.numerator, h.denominator), (n, O/N, h)
    print('SW-1 PASS: Haar moment reproduces the Haar mean purity (2^k+2^(n-k))/(2^n+1) exactly for n = 4..7')
    st = sw.Structure(4); G1 = sw.exact_gamma_map(st, seed=5); G2 = sw.exact_gamma_map(st, seed=11)
    assert all(G1[b][p] == G2[b][p] for b in G1 for p in range(st.NP))
    print('SW-1 PASS: exact gamma map at n = 4 identical for two independent point sets')
    for n in (4, 5): check_genuine(n)

def genuine_P(psi, n, g, flipped=False):
    """<z z z'| G^{(x)n} |z z z'> for an integer real vector psi, z' = psi or eps^{(x)n} psi (exact)."""
    import numpy as np
    z = np.array(psi, dtype=object).reshape((2,)*n); z3 = z
    if flipped:
        for i in range(n):      # eps on every site: (eps v)_0 = v_1, (eps v)_1 = -v_0
            z3 = np.stack([np.take(z3, 1, axis=i), -np.take(z3, 0, axis=i)], axis=i)
    Phi = np.multiply.outer(np.multiply.outer(z, z), z3)                     # axes (c1 sites, c2 sites, c3 sites)
    Phi = Phi.transpose([c*n+i for i in range(n) for c in range(3)]).reshape((8,)*n)
    G = sw.G_of(g); Gm = np.array([[G[i, j] for j in range(8)] for i in range(8)], dtype=object)
    V = Phi
    for i in range(n): V = np.moveaxis(np.tensordot(Gm, V, axes=([1], [i])), 0, i)
    return sum(a*b for a, b in zip(Phi.reshape(-1), V.reshape(-1)))

def check_genuine(n, seed=7):
    """For a random integer real state: fit the parameters x from genuine values P(g) (exact), check the fit
    at further points, and check the flipped genuine values against sum_b tr[Gamma_b(x) R_b(g)]."""
    import numpy as np
    rng = np.random.default_rng(seed); psi = [int(v) for v in rng.integers(-3, 4, size=2**n)]
    st = sw.Structure(n); G = sw.exact_gamma_map(st)
    def rand_g():
        v = [Q(int(x)) for x in rng.integers(-3, 4, size=5)]
        return (v[0], flint.fmpq_mat([[v[1], v[2]], [v[3], v[4]]]))
    pts = [rand_g() for _ in range(st.NP+4)]
    A = flint.fmpq_mat([st.P_params(g) for g in pts]); b = flint.fmpq_mat([[genuine_P(psi, n, g)] for g in pts])
    Asq = flint.fmpq_mat([[A[i, j] for j in range(st.NP)] for i in range(st.NP)]); bsq = flint.fmpq_mat([[b[i, 0]] for i in range(st.NP)])
    x = Asq.solve(bsq)
    assert A*x == b, 'genuine moments not in the parameter family'
    N = {bi: sum((G[bi][p]*x[p, 0] for p in range(st.NP)), flint.fmpq_mat(G[bi][0].nrows(), G[bi][0].nrows())) for bi in G}
    for _ in range(4):
        g = rand_g(); R = st.Rlist('ga', g)
        lhs = sum((sum(((N[bi]*R[bi])[i, i] for i in range(N[bi].nrows())), Q(0)) for bi in N), Q(0))
        assert lhs == genuine_P(psi, n, g, flipped=True)
    mins = min(np.linalg.eigvalsh(np.array([[sw.ffloat(M[i, j]) for j in range(M.ncols())] for i in range(M.nrows())])).min()/max(1e-300, max(abs(sw.ffloat(M[i, i])) for i in range(M.nrows()))) for M in N.values())
    assert mins > -1e-12
    print('SW-1 PASS n=%d: a random integer state lies in the parameter family (%d+4 exact points), its flipped values equal '
          'sum_b tr[Gamma_b(x) R_b(g)] exactly, gamma-blocks PSD (min relative eigenvalue %.1e)' % (n, st.NP, mins))

GAMMA = {}
def certify(key, rec):
    nk = key.split(':'); n = int(nk[0]); k = int(nk[1]) if len(nk) > 1 else n//2
    t0 = time.time(); st = sw.Structure(n); on, oo = st.functionals(k)
    if n not in GAMMA: GAMMA[n] = sw.exact_gamma_map(st)    # one exact map per n, shared by its keys
    G = GAMMA[n]
    lam = Fr(rec['lam']); Z = sw.gamma_multipliers(st, G, rec)
    assert set(Z) <= set(G) and all(Z[b] == Z[b].transpose() and sw.is_pd(Z[b]) for b in Z)
    print('SW-2 PASS n=%d: %d parameters, %d omega- and %d gamma-blocks; exact gamma map; %d multipliers positive definite (%.0fs)'
          % (n, st.NP, len(st.blocks['om']), len(st.blocks['ga']), len(Z), time.time()-t0), flush=True)
    c = sw.gamma_adjoint(st, G, Z)
    D = sw.omega_residual(st, on, oo, Q(lam.numerator, lam.denominator), c)
    assert all(sw.is_pd(M) for M in D)
    print('SW-3 PASS n=%d: all %d omega-blocks of the residual positive definite (%.0fs)' % (n, len(D), time.time()-t0))
    print('SW-4 n=%d: every pure %d-qubit state has mean %d-body purity >= %s = %.10f' % (n, n, k, lam, float(lam)), flush=True)
    if len(nk) > 1 and lam > Fr(1, 2**k):
        print('SW-4 n=%d: %s > 2^-%d, so no pure %d-qubit state is %d-uniform: no ((%d,1,%d))_2 code' % (n, lam, k, n, k, n, k+1), flush=True)

if __name__ == '__main__':
    assert sw.check_site(); print('SW-0 PASS: per-site algebra, Fourier inversion, flip involution, antisymmetriser = 0')
    check_sw1()
    data = json.load(open(os.path.join(HERE, '..', 'data', 'three_copy_sw_certificates.json')))
    keys = sys.argv[1].split(',') if len(sys.argv) > 1 else sorted(data, key=lambda s: tuple(map(int, s.split(':'))))
    for key in keys: certify(key, data[key])
