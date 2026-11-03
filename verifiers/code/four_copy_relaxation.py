"""Four-copy spin-flip relaxation of the mean balanced purity (numerical construction).

Statement: sec:fourcopy (in four_copy_bound.tex)
Imported by explore_seven_qubit_four_copy_sdp.py.

For a unit z in (C^2)^{(x)n} the per-site U(2)^{(x)4}-twirl X of (zz^*)^{(x)4} lies in A^{(x)n}, where
A = image of C[S_4] in End((C^2)^{(x)4}) = C + M_3 + M_2 (S_4 irreps [4], [3,1], [2,2]; the others
vanish on qubits).  X is positive, S_n-symmetric, real after z <-> conj(z) averaging, and fixed by
global copy permutations on both sides, so its Fourier block of each site-type pattern is supported on
S_4-invariant vectors.  Writing those blocks in symmetry-adapted bases (copy invariants, Young-subgroup
invariants, S_a x S_b isotypic components) leaves a small number of free parameters (167 at n = 7).
The flipped moments Y_1, Y_2 (copy 4, resp. copies 3 and 4, replaced by eps^{(x)n} conj(z)) are the
images of X under per-site linear maps T1, T2 (partial transpose with eps conjugation), and are
positive with S_3-, resp. S_2 x S_2-invariant support.  Everything is in floating point; site
coordinates are (t,i,j), t in {0:[4], 1:[3,1], 2:[2,2]}, matrix units of real orthogonal irreps.
"""
import itertools, math, os, pickle, time
from functools import lru_cache
import numpy as np
import scipy.linalg

PERMS = list(itertools.permutations(range(4)))
IDX = {p: k for k, p in enumerate(PERMS)}
def comp(a, b): return tuple(a[b[i]] for i in range(4))
MUL = np.array([[IDX[comp(a, b)] for b in PERMS] for a in PERMS])
INV = np.array([IDX[tuple(int(v) for v in np.argsort(p))] for p in PERMS])
def op16(p):
    M = np.zeros((16, 16))
    for idx in itertools.product(range(2), repeat=4):
        out = [0]*4
        for i in range(4): out[p[i]] = idx[i]
        M[int(''.join(map(str, out)), 2), int(''.join(map(str, idx)), 2)] = 1
    return M
OP = np.array([op16(p) for p in PERMS])
def pm(p, k):
    P = np.zeros((k, k))
    for i, j in enumerate(p): P[j, i] = 1
    return P
def sumzero_basis(k):
    B = np.linalg.qr(np.vstack([np.ones(k), np.eye(k)[:-1]]).T)[0][:, 1:]
    return B
V4 = sumzero_basis(4); V3 = sumzero_basis(3)
PAIRINGS = [frozenset({frozenset({0, 1}), frozenset({2, 3})}), frozenset({frozenset({0, 2}), frozenset({1, 3})}),
            frozenset({frozenset({0, 3}), frozenset({1, 2})})]
def pairing_perm(p):
    return tuple(PAIRINGS.index(frozenset(frozenset(p[x] for x in e) for e in P)) for P in PAIRINGS)
RHO = {0: np.ones((24, 1, 1)),
       1: np.array([V4.T @ pm(p, 4) @ V4 for p in PERMS]),
       2: np.array([V3.T @ pm(pairing_perm(p), 3) @ V3 for p in PERMS])}
DIM = {0: 1, 1: 3, 2: 2}          # Specht dims
UDIM = {0: 5, 1: 3, 2: 1}         # U(2) irrep dims (spin 2, 1, 0)
# coordinates of the 14-dim algebra: (t,i,j)
COORDS = [(t, i, j) for t in (0, 1, 2) for i in range(DIM[t]) for j in range(DIM[t])]
NC = len(COORDS)
def fourier(a):
    """Fourier components (14-vector) of an operator a in the span of OP."""
    c, res, *_ = np.linalg.lstsq(OP.reshape(24, -1).T, a.reshape(-1), rcond=None)
    assert np.abs(OP.reshape(24, -1).T @ c - a.reshape(-1)).max() < 1e-10, 'not in algebra'
    return np.array([sum(c[k]*RHO[t][k][i, j] for k in range(24)) for (t, i, j) in COORDS])
def embed(t, i, j):
    return sum(DIM[t]/24*RHO[t][k][i, j]*OP[k] for k in range(24))
EPS = np.array([[0., 1.], [-1., 0.]])
def ptrans(M, copies):
    T = M.reshape((2,)*8)
    perm = list(range(8))
    for c in copies: perm[c], perm[4+c] = perm[4+c], perm[c]
    return T.transpose(perm).reshape(16, 16)
def flipmap(copies):
    E = np.eye(1)
    for c in range(4): E = np.kron(E, EPS if c in copies else np.eye(2))
    def phi(M): return E @ ptrans(M, copies) @ E.T
    return np.array([fourier(phi(embed(*co))) for co in COORDS]).T   # column = image of coord
T1 = flipmap([3]); T2 = flipmap([2, 3])
def check():
    for a in range(24):
        for b in range(24):
            c = MUL[a, b]
            assert np.allclose(OP[a] @ OP[b], OP[c])
            for t in (1, 2): assert np.allclose(RHO[t][a] @ RHO[t][b], RHO[t][c])
    for t in (1, 2):
        for a in range(24): assert np.allclose(RHO[t][a] @ RHO[t][a].T, np.eye(DIM[t]))
    for k, co in enumerate(COORDS):
        assert np.allclose(fourier(embed(*co)), np.eye(NC)[k])
    return True

def partitions(n, maxpart=None):
    if maxpart is None: maxpart = n
    if n == 0: yield (); return
    for k in range(min(n, maxpart), 0, -1):
        for p in partitions(n-k, k): yield (k,)+p

@lru_cache(None)
def char(lam, mu):
    """Murnaghan-Nakayama: chi^lam at cycle type mu (tuples)."""
    if not mu: return 1 if sum(lam) == 0 else 0
    k, rest = mu[0], mu[1:]
    L = len(lam); beta = [lam[i]+L-1-i for i in range(L)]; bs = set(beta); tot = 0
    for b in beta:
        nb = b-k
        if nb < 0 or nb in bs: continue
        sign = (-1)**sum(1 for x in beta if nb < x < b)
        newb = sorted([x for x in beta if x != b]+[nb], reverse=True)
        newlam = tuple(newb[i]-(L-1-i) for i in range(L))
        newlam = tuple(x for x in newlam if x > 0)
        tot += sign*char(newlam, rest)
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
    """V: (N, k) with N = prod(dims); permute tensor factors: new factor perm[i] <- old factor i."""
    m = len(dims); k = V.shape[1]
    T = V.reshape(tuple(dims)+(k,))
    # output axis perm[i] holds input axis i
    order = [0]*m
    for i in range(m): order[perm[i]] = i
    return T.transpose(order+[m]).reshape(-1, k)

def copy_invariants(a, b, group):
    """Orthonormal basis of vectors in (R^3)^{(x)a}(x)(R^2)^{(x)b} fixed by the copy subgroup."""
    N = 3**a*2**b
    P = np.zeros((N, N))
    for g in group:
        M = np.ones((1, 1))
        for _ in range(a): M = np.kron(M, RHO[1][g])
        for _ in range(b): M = np.kron(M, RHO[2][g])
        P += M
    P /= len(group)
    w, v = np.linalg.eigh((P+P.T)/2)
    return v[:, w > 0.5]

def young_perms(lam, offset):
    """All permutations of the Young subgroup S_lam on sites offset.. (as list of index maps)."""
    blocks, s = [], offset
    for l in lam: blocks.append(list(range(s, s+l))); s += l
    out = []
    for ps in itertools.product(*[itertools.permutations(bl) for bl in blocks]):
        out.append(ps)
    return blocks, out

def adapted(a, b, group):
    """For type (a [3,1]-sites, b [2,2]-sites): list of (lam, mu, f, Q) with Q (N x m) orthonormal basis of
    E = copy-invariants ∩ Young(lam,mu)-invariants ∩ isotypic(lam,mu)."""
    dims = [3]*a+[2]*b; N = int(np.prod(dims)) if dims else 1
    U = copy_invariants(a, b, group) if dims else np.ones((1, 1))
    out = []
    if U.shape[1] == 0: return out, U
    Hel = [tuple(p)+tuple(a+x for x in q) for p in itertools.permutations(range(a)) for q in itertools.permutations(range(b))]
    for lam in partitions(a, None):
        if len(lam) > 3: continue
        for mu in partitions(b, None):
            if len(mu) > 2: continue
            # Young average
            bl1, _ = young_perms(lam, 0); bl2, _ = young_perms(mu, a)
            Y = U.copy()
            # average over each block's symmetric group sequentially (product of subgroups)
            for bl in bl1+bl2:
                if len(bl) < 2: continue
                acc = np.zeros_like(Y); cnt = 0
                for pp in itertools.permutations(bl):
                    perm = list(range(a+b))
                    for x, y in zip(bl, pp): perm[x] = y
                    acc += site_perm_apply(Y, dims, perm); cnt += 1
                Y = acc/cnt
            uu, ss, _ = np.linalg.svd(Y, full_matrices=False)
            R = uu[:, ss > 1e-9]
            if R.shape[1] == 0: continue
            # isotypic projection with characters of S_a x S_b
            f = fdim(lam)*fdim(mu)
            acc = np.zeros_like(R)
            for p in Hel:
                c = char(lam, cycle_type(p[:a]) if a else ()) * char(mu, cycle_type(tuple(x-a for x in p[a:])) if b else ())
                if c: acc += c*site_perm_apply(R, dims, list(p))
            acc *= f/len(Hel)
            uu, ss, _ = np.linalg.svd(acc, full_matrices=False)
            Q = uu[:, ss > 1e-9]
            if Q.shape[1]: out.append((lam, mu, f, Q))
    return out, U

S4 = list(range(24))
S3 = [k for k, p in enumerate(PERMS) if p[3] == 3]
S2S2 = [k for k, p in enumerate(PERMS) if set(p[:2]) == {0, 1}]

SL = {0: slice(0, 1), 1: slice(1, 10), 2: slice(10, 14)}

def types(n): return [(a, b, n-a-b) for a in range(n+1) for b in range(n+1-a)]
def tlist(tp): a, b, c = tp; return [1]*a+[2]*b+[0]*c

def bases(n, cache=None, log=print):
    """Symmetry-adapted bases for every site-type pattern (a [3,1]-sites, b [2,2]-sites, c [4]-sites)."""
    if cache and os.path.exists(cache): return pickle.load(open(cache, 'rb'))
    out = {}
    for tp in types(n):
        a, b, c = tp
        for name, grp in (('om', S4), ('g1', S3), ('g2', S2S2)):
            ad, U = adapted(a, b, grp)
            out[(tp, name)] = ad
    if cache: pickle.dump(out, open(cache, 'wb'))
    return out

def group_elems(a, b):
    return [tuple(p)+tuple(a+x for x in q) for p in itertools.permutations(range(a)) for q in itertools.permutations(range(b))]

def omega_params(n, B):
    """List of (tp, lam-index, k, l, Cmatrix) spanning symmetric H-invariant matrices on copy-invariants."""
    out = []
    for tp in types(n):
        a, b, c = tp; dims = [3]*a+[2]*b
        for li, (lam, mu, f, Q) in enumerate(B[(tp, 'om')]):
            H = group_elems(a, b)
            if not dims:
                out.append((tp, li, 0, 0, np.ones((1, 1)))); continue
            # choose f group elements giving an invertible Y
            cols = np.hstack([site_perm_apply(Q[:, :1], dims, list(h)) for h in H])
            _, _, piv = scipy.linalg.qr(cols, mode='economic', pivoting=True)
            sel = [H[p] for p in piv[:f]]
            Rk = [np.hstack([site_perm_apply(Q[:, k:k+1], dims, list(h)) for h in sel]) for k in range(Q.shape[1])]
            G = Rk[0].T @ Rk[0]; Gi = np.linalg.inv(G)
            for k in range(Q.shape[1]):
                for l in range(k, Q.shape[1]):
                    C = Rk[k] @ Gi @ Rk[l].T
                    C = (C+C.T)/2
                    out.append((tp, li, k, l, C))
    return out

def orderings(tp):
    tl = tlist(tp); n = len(tl)
    seen = {}
    for p in itertools.permutations(range(n)):
        s = tuple(tl[p.index(i)] for i in range(n))   # site i gets type of sorted position p^{-1}(i)
        if s not in seen: seen[s] = p
    return seen   # s -> perm p: sorted position j goes to site p[j]

def place(Cmat, tp, n, X):
    """Add S_n-symmetrization of the sorted block Cmat (type tp) into dense tensor X (shape (14,)*n)."""
    tl = tlist(tp); d = [DIM[t] for t in tl]
    sub = Cmat.reshape(d+d).transpose([x for j in range(n) for x in (j, n+j)]).reshape([dd*dd for dd in d]) if n else Cmat.reshape(())
    for s, p in orderings(tp).items():
        # sorted axis j -> site p[j]
        order = [0]*n
        for j in range(n): order[p[j]] = j
        X[tuple(SL[t] for t in s)] += sub.transpose(order)

def apply_all(T, X):
    n = X.ndim
    for k in range(n):
        A = NC**k; Bsz = NC**(n-1-k)
        X = np.matmul(T, X.reshape(A, NC, Bsz)).reshape((NC,)*n)
    return X

def extract(X, tp):
    tl = tlist(tp); d = [DIM[t] for t in tl]; n = len(tl)
    sub = X[tuple(SL[t] for t in tl)]
    if n == 0: return sub.reshape(1, 1)
    sub = sub.reshape([x for dd in d for x in (dd, dd)]).transpose([2*j for j in range(n)]+[2*j+1 for j in range(n)])
    N = int(np.prod(d)); return sub.reshape(N, N)

SWAP = PERMS.index((1, 0, 2, 3))
def obj_norm(n, k, params):
    """Linear functionals: normalization tr X and mean k-body purity, per parameter."""
    on, oo = [], []
    for (tp, li, kk, ll, C) in params:
        a, b, c = tp; tl = tlist(tp)
        M = math.factorial(n)//(math.factorial(a)*math.factorial(b)*math.factorial(c))
        D = 3**a*5**c
        on.append(D*M*np.trace(C))
        tot = 0.0
        for S in itertools.combinations(range(n), k):
            K = np.ones((1, 1))
            for i, t in enumerate(tl): K = np.kron(K, RHO[t][SWAP if i in S else 0])
            tot += np.sum(C*K.T)
        oo.append(D*M*tot/math.comb(n, k))
    return np.array(on), np.array(oo)

def gamma_maps(n, B, params, name, T, log=print, cache=None):
    """For each parameter, the compressed gamma blocks: dict (tp,li) -> array (P, m, m)."""
    if cache and os.path.exists(cache): return pickle.load(open(cache, 'rb'))
    keys = [(tp, li) for tp in types(n) for li in range(len(B[(tp, name)]))]
    out = {key: np.zeros((len(params),)+(B[(key[0], name)][key[1]][3].shape[1],)*2) for key in keys}
    t0 = time.time()
    for pi, (tp, li, kk, ll, C) in enumerate(params):
        X = np.zeros((NC,)*n); place(C, tp, n, X)
        Y = apply_all(T, X); del X
        for (tq, lj) in keys:
            Q = B[(tq, name)][lj][3]
            Yb = extract(Y, tq)
            out[(tq, lj)][pi] = Q.T @ Yb @ Q
        del Y
        if pi % 20 == 0: log('  %s: parameter %d/%d mapped (%.0fs)' % (name, pi, len(params), time.time()-t0))
    if cache: pickle.dump(out, open(cache, 'wb'))
    return out

def haar_point(n, params, on):
    """Parameters of the Haar average of the moment, sum_pi pi^{(x)n} (normalised): an interior point."""
    f = np.array([[RHO[t][k][i, j] for (t, i, j) in COORDS] for k in range(24)])
    X = np.zeros((NC,)*n)
    for k in range(24):
        T = f[k]
        for _ in range(n-1): T = np.multiply.outer(T, f[k])
        X += T
    x = np.zeros(len(params))
    for tp in types(n):
        idx = [i for i, P in enumerate(params) if P[0] == tp]
        if not idx: continue
        A = np.array([params[i][4].ravel() for i in idx]).T
        x[idx] = np.linalg.lstsq(A, extract(X, tp).ravel(), rcond=None)[0]
    return x/(on @ x)

def omega_groups(params):
    g = {}
    for i, (tp, li, k, l, C) in enumerate(params): g.setdefault((tp, li), []).append((k, l, i))
    return g

def omega_matrix(x, lst, expr=False):
    """omega block: diagonal parameters x_kk, off-diagonal x_kl/2."""
    m = max(max(k, l) for k, l, _ in lst)+1
    E = [[0]*m for _ in range(m)]
    for k, l, i in lst:
        if k == l: E[k][k] = x[i]
        else: E[k][l] = x[i]/2; E[l][k] = x[i]/2
    if expr:
        import cvxpy as cp
        return cp.bmat([[e if not isinstance(e, int) else cp.Constant(0.0) for e in r] for r in E])
    return np.array(E, dtype=float)

def solve_centred(params, on, oo, gmaps, use=('g1', 'g2'), xh=None, eps=0.0, tol=1e-8):
    """The relaxation with every block congruence-centred at the interior point xh (Cholesky L of the block
    at xh: constraint L^{-1} M(x) L^{-T} >= 0).  With eps > 0 the centred flip blocks' traces are rewarded and
    the returned multipliers Z = L^{-T} (dual + eps I) L^{-1} are strictly positive definite.  Returns
    (value, x, status, [(name, key, Z)]) with Z in the uncentred block coordinates."""
    import cvxpy as cp
    NP = len(params); x = cp.Variable(NP); cons = [on @ x == 1]; pen = 0; gl = []
    for lst in omega_groups(params).values():
        Li = np.linalg.inv(np.linalg.cholesky(omega_matrix(xh, lst)))
        cons.append(Li @ omega_matrix(x, lst, expr=True) @ Li.T >> 0)
    for name in use:
        for key, arr in gmaps[name].items():
            if np.abs(arr).max() < 1e-12: continue
            M0 = np.tensordot(xh, arr, 1); Li = np.linalg.inv(np.linalg.cholesky((M0+M0.T)/2))
            A = np.einsum('ab,pbc,dc->pad', Li, arr, Li)
            M = cp.reshape(x @ A.reshape(NP, -1), arr.shape[1:], order='C'); Ms = (M+M.T)/2
            c = Ms >> 0; cons.append(c); gl.append((name, key, Li, c)); pen = pen+cp.trace(Ms)
    pr = cp.Problem(cp.Minimize(oo @ x - eps*pen), cons)
    pr.solve(solver='CLARABEL', max_iter=1000, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol)
    Zs = [(name, key, Li.T @ (c.dual_value + eps*np.eye(Li.shape[0])) @ Li) for name, key, Li, c in gl]
    return pr.value, x.value, pr.status, Zs
