"""Exact (integer/rational) form of the four-copy spin-flip relaxation and its certificate check.

Statement: sec:fourcopy (in four_copy_bound.tex)
Imported by certify_seven_qubit_four_copy_bound.py and search_four_copy_certificate.py.
The floating-point construction (four_copy_relaxation.py) has the same structure in orthonormal
coordinates; this module redoes it over the integers:

* site algebra A = C + M_3 + M_2 with *integer* irreps of S_4 ([3,1] on columns e0-e1, e1-e2, e2-e3 of
  C^4; [2,2] on the sum-zero vectors of the three pairings), Gram forms GRAM[t] = V^T V, and matrix units
  EMB embedded in C[S_4] by the Schur-orthogonality formula (checked exactly);
* the flip maps T1, T2 (copy 4, resp. copies 3 and 4, partially transposed with eps conjugation) as exact
  rational 14 x 14 matrices; 12*T is integral;
* rational symmetry-adapted bases with integer columns (copy invariants, Young subgroup invariants,
  S_a x S_b isotypic parts), columns chosen as exact mod-p rref pivots, so the bases are reproducible;
* omega parameters F_p = R_k adj(K) R_l^T H + R_l adj(K) R_k^T H (block (tp, li), K = R_0^T H R_0);
* the certificate functional c_p = sum_gamma <Z_gamma, Q'^T H (12 T)^{(x)n} X_p Q'> computed modulo
  primes p < 2^20 in float64 (all intermediate sums exact) and reconstructed by CRT under a rigorous
  integer triangle-inequality bound on |c_p|.
"""
import itertools, math
from fractions import Fraction as Fr
from functools import lru_cache, reduce
import numpy as np, flint

# ---------------------------------------------------------------- one site
PERMS = list(itertools.permutations(range(4)))
IDX = {p: k for k, p in enumerate(PERMS)}
def op16(p):
    M = np.zeros((16, 16), dtype=np.int64)
    for idx in itertools.product(range(2), repeat=4):
        out = [0]*4
        for i in range(4): out[p[i]] = idx[i]
        M[int(''.join(map(str, out)), 2), int(''.join(map(str, idx)), 2)] = 1
    return M
OP = np.array([op16(p) for p in PERMS])
def pm(p, k):
    P = np.zeros((k, k), dtype=np.int64)
    for i, j in enumerate(p): P[j, i] = 1
    return P
V31 = np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 1], [0, 0, -1]])
V22 = np.array([[1, 0], [-1, 1], [0, -1]])
PAIRINGS = [frozenset({frozenset({0, 1}), frozenset({2, 3})}), frozenset({frozenset({0, 2}), frozenset({1, 3})}),
            frozenset({frozenset({0, 3}), frozenset({1, 2})})]
def pairing_perm(p):
    return tuple(PAIRINGS.index(frozenset(frozenset(p[x] for x in e) for e in P)) for P in PAIRINGS)
def intrep(V, P):
    R = np.rint(np.linalg.lstsq(V, P @ V, rcond=None)[0]).astype(np.int64)
    assert (V @ R == P @ V).all(); return R
RHO = {0: np.ones((24, 1, 1), dtype=np.int64),
       1: np.array([intrep(V31, pm(p, 4)) for p in PERMS]),
       2: np.array([intrep(V22, pm(pairing_perm(p), 3)) for p in PERMS])}
GRAM = {0: np.ones((1, 1), dtype=np.int64), 1: V31.T @ V31, 2: V22.T @ V22}
DIM = {0: 1, 1: 3, 2: 2}; UDIM = {0: 5, 1: 3, 2: 1}
COORDS = [(t, i, j) for t in (0, 1, 2) for i in range(DIM[t]) for j in range(DIM[t])]; NC = 14
SL = {0: slice(0, 1), 1: slice(1, 10), 2: slice(10, 14)}
SWAP = PERMS.index((1, 0, 2, 3))

def fourier_coeffs(c):
    """Integer-irrep Fourier coordinates (14 Fractions) of sum_pi c_pi P_pi."""
    return [sum((c[k]*int(RHO[t][k][i, j]) for k in range(24)), Fr(0)) for (t, i, j) in COORDS]
def embed_coeffs():
    """c^{(t,i,j)}_pi = (d_t/24) rho_t(pi^{-1})_{ji}: the element of C[S_4] with Fourier coordinates E^t_ij."""
    INVI = [IDX[tuple(int(v) for v in np.argsort(p))] for p in PERMS]; out = []
    for (t, i, j) in COORDS:
        c = [Fr(DIM[t], 24)*int(RHO[t][INVI[k]][j, i]) for k in range(24)]
        assert fourier_coeffs(c) == [Fr(int(co == (t, i, j))) for co in COORDS]
        out.append(c)
    return out
EMB = embed_coeffs()
EPS = np.array([[0, 1], [-1, 0]], dtype=np.int64)
def ptrans(M, copies):
    T = M.reshape((2,)*8); perm = list(range(8))
    for c in copies: perm[c], perm[4+c] = perm[4+c], perm[c]
    return T.transpose(perm).reshape(16, 16)
def solve_ops(target):
    """Rational c with sum_pi c_pi P_pi = target (16 x 16 object matrix); exactness asserted."""
    A = OP.reshape(24, -1).T.astype(float)
    c = np.linalg.lstsq(A, np.array([float(v) for v in target.reshape(-1)]), rcond=None)[0]
    cf = [Fr(v).limit_denominator(10**4) for v in c]
    for e in range(256):
        assert sum((cf[k]*int(OP[k].reshape(-1)[e]) for k in range(24)), Fr(0)) == target.reshape(-1)[e]
    return cf
def flipmap(copies):
    """Exact 14 x 14 matrix T[kp][k] of a -> (eps on copies) a^{T_copies} (eps on copies)^T on A."""
    E = np.eye(1, dtype=np.int64)
    for c in range(4): E = np.kron(E, EPS if c in copies else np.eye(2, dtype=np.int64))
    cols = []
    for c in EMB:
        O = sum(c[k]*OP[k].astype(object) for k in range(24))
        cols.append(fourier_coeffs(solve_ops(E.astype(object) @ ptrans(O, copies) @ E.T.astype(object))))
    return [[cols[k][kp] for k in range(NC)] for kp in range(NC)]
T1 = flipmap([3]); T2 = flipmap([2, 3])
def Tint(T):
    assert all((12*v).denominator == 1 for row in T for v in row)
    return np.array([[int(12*v) for v in row] for row in T], dtype=np.int64)
T1i, T2i = Tint(T1), Tint(T2)

# ---------------------------------------------------------------- symmetry-adapted bases
def partitions(n, maxpart=None):
    if maxpart is None: maxpart = n
    if n == 0: yield (); return
    for k in range(min(n, maxpart), 0, -1):
        for p in partitions(n-k, k): yield (k,)+p
@lru_cache(None)
def char(lam, mu):
    """Murnaghan-Nakayama: chi^lam at cycle type mu."""
    if not mu: return 1 if sum(lam) == 0 else 0
    k, rest = mu[0], mu[1:]
    L = len(lam); beta = [lam[i]+L-1-i for i in range(L)]; bs = set(beta); tot = 0
    for b in beta:
        nb = b-k
        if nb < 0 or nb in bs: continue
        sign = (-1)**sum(1 for x in beta if nb < x < b)
        newb = sorted([x for x in beta if x != b]+[nb], reverse=True)
        tot += sign*char(tuple(x for x in (newb[i]-(L-1-i) for i in range(L)) if x > 0), rest)
    return tot
def cycle_type(p):
    seen, ct = set(), []
    for i in range(len(p)):
        if i in seen: continue
        l, j = 0, i
        while j not in seen: seen.add(j); j = p[j]; l += 1
        ct.append(l)
    return tuple(sorted(ct, reverse=True))
def fdim(lam): return char(lam, (1,)*sum(lam))
def site_perm_apply(V, dims, perm):
    m = len(dims); k = V.shape[1]; T = V.reshape(tuple(dims)+(k,)); order = [0]*m
    for i in range(m): order[perm[i]] = i
    return T.transpose(order+[m]).reshape(-1, k)
def kronlist(ms):
    M = np.ones((1, 1), dtype=np.int64)
    for x in ms: M = np.kron(M, x)
    return M
def pivots_mod(Y, p=2147483629):
    """Pivot columns of the reduced row echelon form of Y mod p (exact and deterministic)."""
    R, r = flint.nmod_mat(Y.shape[0], Y.shape[1], [int(v) % p for v in Y.ravel()], p).rref()
    piv = []
    for i in range(r):
        piv.append(next(j for j in range(Y.shape[1]) if int(R[i, j]) != 0))
    return piv
def select_cols(Y):
    """Independent integer columns spanning the column space of Y: the mod-p rref pivot columns, gcd-reduced.
    Independence is exact modulo p. Completeness is checked by the character
    dimension identities in adapted(), not by a floating-point rank estimate."""
    if Y.shape[1] == 0: return Y
    piv = pivots_mod(Y)
    out = Y[:, piv].copy()
    for c in range(out.shape[1]): out[:, c] //= reduce(math.gcd, [int(v) for v in out[:, c]])
    return out
def adapted(a, b, group):
    dims = [3]*a+[2]*b
    if dims: U = select_cols(sum(kronlist([RHO[1][g]]*a+[RHO[2][g]]*b) for g in group))
    else: U = np.ones((1, 1), dtype=np.int64)
    # The average over group is an exact projection. Its trace is its rank;
    # modular independence plus this upper bound certifies a complete basis.
    rank_num = sum(int(np.trace(RHO[1][g]))**a * int(np.trace(RHO[2][g]))**b
                   for g in group)
    assert rank_num % len(group) == 0
    assert U.shape[1] == rank_num // len(group)
    out = []
    if U.shape[1] == 0: return out
    Hel = group_elems(a, b)
    for lam in partitions(a):
        if len(lam) > 3: continue
        for mu in partitions(b):
            if len(mu) > 2: continue
            blocks, s = [], 0
            for l in lam+mu: blocks.append(list(range(s, s+l))); s += l
            Y = U.copy()
            for bl in blocks:
                if len(bl) < 2: continue
                acc = np.zeros_like(Y)
                for pp in itertools.permutations(bl):
                    perm = list(range(a+b))
                    for x, y in zip(bl, pp): perm[x] = y
                    acc += site_perm_apply(Y, dims, perm)
                Y = select_cols(acc)
                if Y.shape[1] == 0: break
            if Y.shape[1] == 0: continue
            acc = np.zeros_like(Y)
            for p in Hel:
                c = char(lam, cycle_type(p[:a]) if a else ())*char(mu, cycle_type(tuple(x-a for x in p[a:])) if b else ())
                if c: acc += c*site_perm_apply(Y, dims, list(p))
            Q = select_cols(acc)
            if Q.shape[1]: out.append((lam, mu, fdim(lam)*fdim(mu), Q))
    assert sum(f*Q.shape[1] for (_, _, f, Q) in out) == U.shape[1]
    return out
S4 = list(range(24)); S3 = [k for k, p in enumerate(PERMS) if p[3] == 3]
S2S2 = [k for k, p in enumerate(PERMS) if set(p[:2]) == {0, 1}]
def types(n): return [(a, b, n-a-b) for a in range(n+1) for b in range(n+1-a)]
def tlist(tp): a, b, c = tp; return [1]*a+[2]*b+[0]*c
def group_elems(a, b):
    return [tuple(p)+tuple(a+x for x in q) for p in itertools.permutations(range(a)) for q in itertools.permutations(range(b))]
def bases(n):
    return {(tp, name): adapted(tp[0], tp[1], grp) for tp in types(n) for name, grp in (('om', S4), ('g1', S3), ('g2', S2S2))}

# ---------------------------------------------------------------- block layout
def orderings(tp):
    tl = tlist(tp); n = len(tl); seen = {}
    for p in itertools.permutations(range(n)):
        s = tuple(tl[p.index(i)] for i in range(n))
        if s not in seen: seen[s] = p
    return seen
def to_sub(Mat, tp):
    tl = tlist(tp); d = [DIM[t] for t in tl]; n = len(tl)
    if n == 0: return Mat.reshape(())
    return Mat.reshape(d+d).transpose([x for j in range(n) for x in (j, n+j)]).reshape([x*x for x in d])
def from_sub(sub, tp):
    tl = tlist(tp); d = [DIM[t] for t in tl]; n = len(tl)
    if n == 0: return sub.reshape(1, 1)
    N = int(np.prod(d))
    return sub.reshape([x for dd in d for x in (dd, dd)]).transpose([2*j for j in range(n)]+[2*j+1 for j in range(n)]).reshape(N, N)
def sitewise(Mlist, X, mod=None):
    """(x)_i M_i applied to the rows of X (N x f); float64, exact while entries stay below 2^53."""
    if not mod:
        bound = max((abs(int(v)) for v in X.ravel()), default=0)
        for M in Mlist:
            bound *= max(sum(abs(int(v)) for v in row) for row in M)
        assert bound < 2**53, 'integer tensor contraction would round'
    T = X.reshape([M.shape[1] for M in Mlist]+[X.shape[1]]).astype(float)
    for i, M in enumerate(Mlist):
        T = np.moveaxis(np.tensordot(M.astype(float), T, axes=([1], [i])), 0, i)
        if mod: T = np.fmod(T, mod)
    return T.reshape(-1, X.shape[1])

# ---------------------------------------------------------------- omega parameters, objective, normalization
def omega_data(n, B):
    """Per omega block: R_k (int64 N x f), det K > 0 and adj K (python ints), K = R_0^T H R_0."""
    blocks = []
    for tp in types(n):
        a, b, c = tp; dims = [3]*a+[2]*b; tl = tlist(tp)
        for li, (lam, mu, f, Q) in enumerate(B[(tp, 'om')]):
            if not dims:
                blocks.append(dict(tp=tp, li=li, m=1, f=1, R=[np.ones((1, 1), dtype=np.int64)], det=1, adj=[[1]], triv=True)); continue
            Hel = group_elems(a, b)
            cols = np.hstack([site_perm_apply(Q[:, :1], dims, list(h)) for h in Hel])
            sel = [Hel[p] for p in pivots_mod(cols)]; assert len(sel) == f
            R = [np.hstack([site_perm_apply(Q[:, k:k+1], dims, list(h)) for h in sel]).astype(np.int64) for k in range(Q.shape[1])]
            HR = np.rint(sitewise([GRAM[t] for t in tl], R[0])).astype(np.int64)
            Kf = flint.fmpz_mat([[int(v) for v in row] for row in R[0].T.astype(object).dot(HR.astype(object))])
            d = int(Kf.det()); assert d > 0
            inv = Kf.inv(); adj = [[int(inv[i, j]*d) for j in range(f)] for i in range(f)]
            assert all((inv[i, j]*d).q == 1 for i in range(f) for j in range(f))
            blocks.append(dict(tp=tp, li=li, m=Q.shape[1], f=f, R=R, det=d, adj=adj, triv=False))
    params = [(bi, k, l) for bi, bl in enumerate(blocks) for k in range(bl['m']) for l in range(k, bl['m'])]
    return blocks, params
def pair(bl, k, l, Gfun, p=None):
    """<F_p, G> with F_p as above and G given by its action on N x f matrices (exact ints, or mod p)."""
    adj = bl['adj']
    def tr(Ra, Rb):
        if p is None: M = Ra.T.astype(object).dot(Gfun(Rb).astype(object))
        else: M = np.fmod((Ra % p).T.astype(float) @ Gfun(Rb), p)
        s = sum(adj[i][j]*int(M[j, i]) for i in range(len(adj)) for j in range(len(adj)))
        return s if p is None else s % p
    if bl['triv']: return tr(bl['R'][0], bl['R'][0])
    s = tr(bl['R'][l], bl['R'][k]) + tr(bl['R'][k], bl['R'][l])
    return s if p is None else s % p
def obj_norm(n, kk, blocks, params):
    """Exact normalization tr X and mean kk-body purity of the omega parameters (rational)."""
    on, oo = [], []
    for (bi, k, l) in params:
        bl = blocks[bi]; a, b, c = bl['tp']; tl = tlist(bl['tp'])
        mult = math.factorial(n)//(math.factorial(a)*math.factorial(b)*math.factorial(c))*3**a*5**c
        on.append(Fr(mult*pair(bl, k, l, lambda X: np.rint(sitewise([GRAM[t] for t in tl], X)).astype(np.int64))))
        tot = 0
        for S in itertools.combinations(range(n), kk):
            G = [GRAM[t] @ RHO[t][SWAP if i in S else 0] for i, t in enumerate(tl)]
            tot += pair(bl, k, l, lambda X: np.rint(sitewise(G, X)).astype(np.int64))
        oo.append(Fr(mult*tot, math.comb(n, kk)))
    return on, oo

# ---------------------------------------------------------------- certificate functional
def _adjoint(n, Bx, Zint, keys, p):
    """sum over flip families of the symmetrized V = (12 T^T)^{(x)n} W, W = sum_gamma H Q' Z Q'^T (mod p, or |.| bound if p is None)."""
    f = (lambda A: np.fmod(A, p)) if p else (lambda A: A)
    acc = {tp: None for tp in types(n)}
    for name, T in (('g1', T1i), ('g2', T2i)):
        W = np.zeros((NC,)*n)
        for (nm, (tp, li)), Z in zip(keys, Zint):
            if nm != name: continue
            Q = Bx[(tp, name)][li][3]; tl = tlist(tp)
            if p:
                HQ = f(sitewise([GRAM[t] for t in tl], Q % p, mod=p)); Zm = np.array([[int(v) % p for v in r] for r in Z], dtype=float)
                Wb = f(f(HQ @ Zm) @ (Q % p).T.astype(float))
            else:
                Qa = np.abs(Q.astype(float)); Wb = sitewise([np.abs(GRAM[t]) for t in tl], Qa) @ np.abs(np.array(Z, dtype=float)) @ Qa.T
            W[tuple(SL[t] for t in tl)] = f(W[tuple(SL[t] for t in tl)] + to_sub(Wb, tp))
        Tt = T.T.astype(float) if p else np.abs(T.T).astype(float)
        for k in range(n): W = f(np.matmul(Tt, W.reshape(NC**k, NC, NC**(n-1-k))).reshape((NC,)*n))
        for tp in types(n):
            s = None
            for o, perm in orderings(tp).items():
                sub = W[tuple(SL[t] for t in o)].transpose(list(perm)); s = sub if s is None else f(s+sub)
            Vs = from_sub(s, tp); acc[tp] = Vs if acc[tp] is None else f(acc[tp]+Vs)
    return acc
def c_mod(n, Bx, blocks, params, Zint, keys, p):
    """c_p mod p for every omega parameter."""
    # Every dot product of reduced residues has at most this many terms.
    width = max([3**n] + [len(Z) for Z in Zint])
    assert width*(p-1)**2 < 2**53
    assert NC*(p-1)*max(abs(int(v)) for T in (T1i,T2i) for v in T.ravel()) < 2**53
    acc = _adjoint(n, Bx, Zint, keys, p); G = {}; out = []
    for (bi, k, l) in params:
        bl = blocks[bi]; tp = bl['tp']; tl = tlist(tp)
        if tp not in G: G[tp] = np.fmod(sitewise([GRAM[t] for t in tl], acc[tp].T.copy(), mod=p), p)
        out.append(pair(bl, k, l, lambda X: np.fmod(G[tp] @ (X % p).astype(float), p), p))
    return out
def c_bound(n, Bx, blocks, params, Zint, keys):
    """Integer triangle-inequality bound for CRT uniqueness; no rounded bounds.

    Bound HQ Z Q^T entrywise, then the n local transpose-flip maps,
    at most 3^n type orderings, and finally the two R/adj(K) contractions.
    This is deliberately coarse, but every operation is a Python integer.
    """
    gram_l1 = {t:max(sum(abs(int(v)) for v in row) for row in GRAM[t]) for t in GRAM}
    init = 0
    for (name, (tp, li)), Z in zip(keys, Zint):
        Q = Bx[(tp, name)][li][3]
        qmax = max(abs(int(v)) for v in Q.ravel())
        zmax = max(abs(int(v)) for row in Z for v in row)
        init += len(Z)**2*qmax**2*zmax*math.prod(gram_l1[t] for t in tlist(tp))
    flip_l1 = max(sum(abs(int(v)) for v in row) for T in (T1i,T2i) for row in T.T)
    acc_bound = init*flip_l1**n*3**n
    out = []
    for bl in blocks:
        rmax = max(abs(int(v)) for R in bl['R'] for v in R.ravel())
        adjmax = max(abs(int(v)) for row in bl['adj'] for v in row)
        N = bl['R'][0].shape[0]
        out.append(2*N**2*bl['f']**2*rmax**2*adjmax
                   *math.prod(gram_l1[t] for t in tlist(bl['tp']))*acc_bound)
    return max(out)
PRIMES = [p for p in range(2**20-1, 2**19, -2) if all(p % q for q in range(3, int(p**0.5)+1, 2))][:60]
def crt(residues, primes):
    M, x = 1, 0
    for r, p in zip(residues, primes): x += M*(((r-x)*pow(M, -1, p)) % p); M *= p
    return x if x <= M//2 else x-M
def c_exact(n, Bx, blocks, params, Zint, keys, log=None):
    bnd = c_bound(n, Bx, blocks, params, Zint, keys); k, M = 0, 1
    while M <= 2*bnd: M *= PRIMES[k]; k += 1
    if log: log('|c_p| <= %.3e: CRT over %d primes' % (bnd, k))
    res = [c_mod(n, Bx, blocks, params, Zint, keys, p) for p in PRIMES[:k]]
    return [crt([r[i] for r in res], PRIMES[:k]) for i in range(len(params))]

# ---------------------------------------------------------------- final check
def leading_minors_positive(E):
    """Sylvester's criterion by fraction-free (Bareiss) elimination on a symmetric Fraction matrix."""
    assert all(len(row) == len(E) for row in E)
    assert all(E[i][j] == E[j][i] for i in range(len(E)) for j in range(len(E)))
    den = reduce(lambda a, b: a*b//math.gcd(a, b), [v.denominator for r in E for v in r], 1)
    A = [[int(v*den) for v in r] for r in E]; m = len(A); prev = 1
    for k in range(m):
        if A[k][k] <= 0: return False
        for i in range(k+1, m):
            for j in range(k+1, m): A[i][j] = (A[i][j]*A[k][k]-A[i][k]*A[k][j])//prev
        prev = A[k][k]
    return True
def exact_check(blocks, params, on, oo, lam, c, scale):
    """d_p = oo_p - lam on_p - c_p/scale; each omega block [d_kl] must be positive definite."""
    d = [oo[i]-lam*on[i]-Fr(c[i], scale) for i in range(len(params))]
    E = {}
    for i, (bi, k, l) in enumerate(params):
        m = blocks[bi]['m']; E.setdefault(bi, [[Fr(0)]*m for _ in range(m)]); E[bi][k][l] = E[bi][l][k] = d[i]
    ok = all(leading_minors_positive(M) for M in E.values())
    return ok, min(np.linalg.eigvalsh(np.array([[float(v) for v in r] for r in M])).min() for M in E.values())
