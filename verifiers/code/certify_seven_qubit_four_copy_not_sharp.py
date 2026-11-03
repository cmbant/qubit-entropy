#!/usr/bin/env python3
"""NS-0--NS-4: exact primal certificate that the four-copy relaxation at n = 7 is not sharp, L_4 < 121/917.

Statement: prop:fourcopy-not-sharp (in four_copy_bound.tex)
The relaxation is the one of thm:fourcopy, in the exact coordinates of four_copy_exact.py: 167 omega
parameters x, tr X = on(x) = 1, objective oo(x) (mean three-body purity), omega blocks
omega_b(x) >= 0 (omega_kk = x_kk, omega_kl = x_kl/2) and flip blocks gamma_g(x) = Q_g^T H B_j(x) Q_g >= 0,
B_j the Fourier block of Y_j = T_j^{(x)7} X.  The data file data/four_copy_primal_n7.json holds an integer
vector x (a floating-point optimum of the Haar-centred relaxation mixed with the Haar point and rounded;
see its 'source' field).  Everything is homogeneous except on(x) = 1, so x/on(x) is feasible as soon as
every block of x is positive semidefinite.

NS-0 regression (not used in the conclusion): the modular forward map x -> 12^n gamma_g(x) equals a direct
     floating-point forward computation at n = 3, 4 for random integer x;
NS-1 the data: 167 integer coordinates in the parameter order of four_copy_exact.omega_data, on(x) > 0, and
     the exact objective oo(x)/on(x) < 121/917;
NS-2 every omega block (78) positive definite (exact fraction-free Sylvester test);
NS-3 every flip block (271) of Y_1, Y_2: 12^7 gamma_g(x) recovered as an integer matrix by CRT under a
     rigorous integer magnitude bound, exactly symmetric, positive definite (exact Sylvester test);
NS-4 conclusion: L_4 <= oo(x)/on(x) < 6597/50000 < 121/917.
Runtime is dominated by NS-3 (one pass over the 14^7 moment tensor per prime and flip); FOURCOPY_PROCS
(default min(2, cores)) sets the number of worker processes, about 3 GB each.
"""
import json, math, os, sys, time
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import four_copy_exact as fx

DATA = os.path.join(HERE, '..', 'data', 'four_copy_primal_n7.json')
FLIPS = (('g1', fx.T1i), ('g2', fx.T2i))
BOUND = Fr(6597, 50000)                     # the stated rational upper bound on L_4


def primes_below(top, count):
    out, p = [], top - 1 if top % 2 == 0 else top - 2
    while len(out) < count:
        if all(p % q for q in range(3, int(p**0.5)+1, 2)): out.append(p)
        p -= 2
    return out
PRIMES = primes_below(2**20, 200)


def omega_W(bl, entries):
    """W = 2 omega: W_kk = 2 x_kk, W_kl = x_kl (python ints)."""
    m = bl['m']; W = [[0]*m for _ in range(m)]
    for i, k, l, v in entries:
        if k == l: W[k][k] = 2*v
        else: W[k][l] = W[l][k] = v
    return W


def by_block(blocks, params, x):
    out = {bi: [] for bi in range(len(blocks))}
    for i, (bi, k, l) in enumerate(params): out[bi].append((i, k, l, x[i]))
    return out


def prepare(n, B, blocks):
    """Exact integer products fixed by the bases: per omega block Ra = [R_0 .. R_{m-1}] and HRa = H Ra,
    per flip block HQ = H Q (H = Gram form of the type; sitewise asserts exact float64 integers)."""
    P = {'n': n, 'R': {}, 'Q': {}}
    for bi, bl in enumerate(blocks):
        if bl['triv']: continue
        Ra = np.hstack(bl['R']).astype(np.int64)
        HRa = np.rint(fx.sitewise([fx.GRAM[t] for t in fx.tlist(bl['tp'])], Ra)).astype(np.int64)
        P['R'][bi] = (Ra, HRa)
    for name, _ in FLIPS:
        for tp in fx.types(n):
            for li, (_, _, _, Q) in enumerate(B[(tp, name)]):
                Q = Q.astype(np.int64)
                P['Q'][(name, tp, li)] = (Q, np.rint(fx.sitewise([fx.GRAM[t] for t in fx.tlist(tp)], Q)).astype(np.int64))
    return P


def moment_mod(P, blocks, params, x, p):
    """The S_n-symmetric moment tensor X(x) (shape (14,)*n) modulo p, float64 with exact residues.
    Per omega block X_tp = R (W (x) adj K) R^T H (= sum_p x_p F_p of four_copy_exact); trivial block x."""
    n = P['n']; X = np.zeros((fx.NC,)*n); per_type = {}; grp = by_block(blocks, params, x)
    for bi, bl in enumerate(blocks):
        tp = bl['tp']
        if bl['triv']:
            (i, k, l, v), = grp[bi]; M = np.array([[v % p]], dtype=float)
        else:
            m, f = bl['m'], bl['f']; assert m*f*(p-1)**2 < 2**53
            W = np.array([[v % p for v in r] for r in omega_W(bl, grp[bi])], dtype=float)
            A = np.array([[v % p for v in r] for r in bl['adj']], dtype=float)
            WA = np.fmod(np.kron(W, A), p)
            Ra, HRa = ((R % p).astype(float) for R in P['R'][bi])
            M = np.fmod(np.fmod(Ra @ WA, p) @ HRa.T, p)          # R (W (x) A) R^T H, H symmetric
        per_type[tp] = M if tp not in per_type else np.fmod(per_type[tp]+M, p)
    for tp, M in per_type.items():
        sub = fx.to_sub(M, tp)
        for s, perm in fx.orderings(tp).items():
            order = [0]*n
            for j in range(n): order[perm[j]] = j
            X[tuple(fx.SL[t] for t in s)] = sub.transpose(order)
    return X


def gamma_mod(P, blocks, params, x, p):
    """{(flip, tp, li): 12^n gamma_g(x) = (HQ)^T B_j(12^n x) Q mod p} for every flip block."""
    n = P['n']; assert 3**n*(p-1)**2 < 2**53 and fx.NC*(p-1)**2 < 2**53
    X = moment_mod(P, blocks, params, x, p); out = {}
    for name, Ti in FLIPS:
        # n steps of Y[rest, i'] = sum_i Y[i, rest] T[i', i]: each applies T to the leading axis and
        # moves it last, so after n steps every axis has been mapped once and the order is restored.
        Tt = (Ti % p).astype(float).T; Y = X.reshape(fx.NC, -1)
        for k in range(n):
            Y = Y.T @ Tt; np.fmod(Y, p, out=Y); Y = Y.reshape(fx.NC, -1)
        Y = Y.reshape((fx.NC,)*n)
        for tp in fx.types(n):
            if not B_KEYS.get((tp, name)): continue
            Yb = fx.from_sub(Y[tuple(fx.SL[t] for t in fx.tlist(tp))], tp)
            for li in range(B_KEYS[(tp, name)]):
                Q, HQ = ((A % p).astype(float) for A in P['Q'][(name, tp, li)])
                out[(name, tp, li)] = np.fmod(HQ.T @ np.fmod(Yb @ Q, p), p)
        del Y
    return out


def gamma_bound(P, blocks, params, x):
    """Rigorous integer bound on every entry of 12^n gamma_g(x) (triangle inequality, python ints)."""
    n = P['n']; grp = by_block(blocks, params, x); xb = {}
    amx = lambda A: max(abs(int(v)) for v in np.asarray(A).ravel())
    for bi, bl in enumerate(blocks):
        tp = bl['tp']
        if bl['triv']: b = max(abs(v) for (_, _, _, v) in grp[bi])
        else:
            Ra, HRa = P['R'][bi]; mf = Ra.shape[1]
            wmax = max(abs(v) for r in omega_W(bl, grp[bi]) for v in r)
            b = mf**2*amx(Ra)*amx(HRa)*wmax*max(abs(v) for r in bl['adj'] for v in r)
        xb[tp] = xb.get(tp, 0)+b
    L = max(sum(abs(int(v)) for v in row) for _, T in FLIPS for row in T)
    ymax = max(xb.values())*L**n
    return max(Q.shape[0]**2*amx(Q)*amx(HQ)*ymax for Q, HQ in P['Q'].values())


B_KEYS = {}; _WORK = None
def _worker(p):
    P, blocks, params, x = _WORK
    return gamma_mod(P, blocks, params, x, p)


def gamma_exact(n, B, blocks, params, x, log=None, procs=1):
    global _WORK
    B_KEYS.clear(); B_KEYS.update({k: len(v) for k, v in B.items() if k[1] in ('g1', 'g2')})
    P = prepare(n, B, blocks); bnd = gamma_bound(P, blocks, params, x); k, M = 0, 1
    while M <= 2*bnd:
        assert k < len(PRIMES), 'prime list exhausted'
        M *= PRIMES[k]; k += 1
    assert M > 2*bnd and len(set(PRIMES[:k])) == k
    if log: log('  |12^%d gamma entries| <= 2^%d < M/2, M = product of %d distinct primes < 2^20 (2^%d); %d process(es)'
                % (n, bnd.bit_length(), k, M.bit_length(), procs))
    _WORK = (P, blocks, params, x)
    if procs > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(procs) as pool: res = pool.map(_worker, PRIMES[:k], chunksize=1)
    else: res = [_worker(p) for p in PRIMES[:k]]
    return {key: [[fx.crt([int(r[key][i, j]) for r in res], PRIMES[:k]) for j in range(res[0][key].shape[1])]
                  for i in range(res[0][key].shape[0])] for key in res[0]}


def pd(M):
    return fx.leading_minors_positive([[Fr(v) for v in r] for r in M])


# ---------------------------------------------------------------- NS-0 regression
def forward_float(n, B, blocks, params, x):
    X = np.zeros((fx.NC,)*n); grp = by_block(blocks, params, x); per_type = {}
    for bi, bl in enumerate(blocks):
        tp = bl['tp']
        if bl['triv']: M = np.array([[float(grp[bi][0][3])]])
        else:
            H = fx.kronlist([fx.GRAM[t] for t in fx.tlist(tp)]).astype(float); A = np.array(bl['adj'], dtype=float)
            M = 0
            for (_, k, l, v) in grp[bi]:
                Rk, Rl = bl['R'][k].astype(float), bl['R'][l].astype(float)
                M = M + v*(Rk @ A @ Rl.T @ H + Rl @ A @ Rk.T @ H)
        per_type[tp] = M + per_type.get(tp, 0)
    for tp, M in per_type.items():
        sub = fx.to_sub(M, tp)
        for s, perm in fx.orderings(tp).items():
            order = [0]*n
            for j in range(n): order[perm[j]] = j
            X[tuple(fx.SL[t] for t in s)] += sub.transpose(order)
    out = {}
    for name, Ti in FLIPS:
        Y = X
        for k in range(n): Y = np.moveaxis(np.tensordot(Ti.astype(float), Y, axes=([1], [k])), 0, k)
        for tp in fx.types(n):
            H = fx.kronlist([fx.GRAM[t] for t in fx.tlist(tp)]).astype(float)
            for li, (_, _, _, Q) in enumerate(B[(tp, name)]):
                Qf = Q.astype(float); out[(name, tp, li)] = Qf.T @ H @ fx.from_sub(Y[tuple(fx.SL[t] for t in fx.tlist(tp))], tp) @ Qf
    return out


if __name__ == '__main__':
    t0 = time.time(); rng = np.random.default_rng(11); worst = 0.0
    for n in (3, 4):
        B = fx.bases(n); bl, pa = fx.omega_data(n, B)
        xr = [int(v) for v in rng.integers(-5, 6, size=len(pa))]
        ge = gamma_exact(n, B, bl, pa, xr); gf = forward_float(n, B, bl, pa, xr)
        assert set(ge) == set(gf)
        for key in gf:
            worst = max(worst, np.abs(np.array(ge[key], dtype=float)-gf[key]).max()/max(1, np.abs(gf[key]).max()))
    assert worst < 1e-9, worst
    print('NS-0 PASS (regression): modular forward map + CRT equals the direct float forward map '
          '(n = 3, 4, random integer x), max relative difference %.1e' % worst, flush=True)

    data = json.load(open(DATA)); n = data['n']; assert n == 7
    B = fx.bases(n); blocks, params = fx.omega_data(n, B)
    print('  n=7 rational bases built (%.0fs)' % (time.time()-t0), flush=True)
    keys = [(blocks[bi]['tp'], blocks[bi]['li'], k, l) for (bi, k, l) in params]
    assert [(tuple(r[0]), r[1], r[2], r[3]) for r in data['keys']] == keys
    x = [int(v) for v in data['x']]; assert len(x) == len(params) == 167
    on, oo = fx.obj_norm(n, 3, blocks, params)
    nrm = sum(a*b for a, b in zip(on, x)); val = sum(a*b for a, b in zip(oo, x))/nrm
    assert nrm > 0 and val < BOUND < Fr(121, 917)
    print('NS-1 PASS: 167 integer coordinates (largest %d bits); on(x) > 0; exact objective oo(x)/on(x) = %.12f '
          '(rational, denominator %d digits) < %s < 121/917 = %.12f; gap to 121/917 %.4e'
          % (max(abs(v).bit_length() for v in x), float(val), len(str(val.denominator)), BOUND, 121/917,
             float(Fr(121, 917)-val)), flush=True)
    grp = by_block(blocks, params, x)
    om_ok = [pd(omega_W(bl, grp[bi]) if not bl['triv'] else [[grp[bi][0][3]]]) for bi, bl in enumerate(blocks)]
    assert all(om_ok) and len(om_ok) == 78
    print('NS-2 PASS: all %d omega blocks positive definite (exact Sylvester)' % len(om_ok), flush=True)
    procs = int(os.environ.get('FOURCOPY_PROCS', min(2, os.cpu_count() or 1)))   # ~3 GB per process
    G = gamma_exact(n, B, blocks, params, x, log=lambda s: print(s, flush=True), procs=procs)
    allkeys = [(nm, tp, li) for nm, _ in FLIPS for tp in fx.types(n) for li in range(len(B[(tp, nm)]))]
    assert sorted(G) == sorted(allkeys) and len(G) == 271
    assert all(M[i][j] == M[j][i] for M in G.values() for i in range(len(M)) for j in range(i))
    gam_ok = [pd(G[k]) for k in allkeys]; assert all(gam_ok)
    sizes = [len(G[k]) for k in allkeys]
    rel = min(np.linalg.eigvalsh(np.array(M, dtype=float)).min()/np.abs(np.array(M, dtype=float)).max() for M in G.values())
    print('NS-3 PASS: all %d flip blocks (g1 %d, g2 %d; sizes %d..%d) exactly symmetric and positive definite '
          '(exact Sylvester; smallest eigenvalue/max entry %.2e, diagnostic only)'
          % (len(G), sum(k[0] == 'g1' for k in G), sum(k[0] == 'g2' for k in G), min(sizes), max(sizes), rel), flush=True)
    print('NS-4 PASS: x/on(x) is feasible for the four-copy relaxation, so L_4 <= %.12f < %s < 121/917; '
          '121/917 - %s = %s  (%.0fs)' % (float(val), BOUND, BOUND, Fr(121, 917)-BOUND, time.time()-t0), flush=True)
