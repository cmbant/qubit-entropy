"""Shared exact construction of the three-copy spin-flip relaxation (TCB-0).

Imported by certify_seven_qubit_three_copy_bound.py and
explore_seven_qubit_three_copy_sdp.py.  Statement: sec:threecopy (in replica_hierarchy.tex)

For a unit z in (C^2)^{(x)n} and a word w = (pi_1..pi_n) of permutations of
three copies, mu(w) = <z^{(x)3}| (x)_i pi_i |z^{(x)3}>.  The relaxation
variable is the real part of mu averaged over site permutations S_n; it is
constant on orbits of words under S_n, global left/right copy permutations and
inversion.  Two families of positive blocks are linear in mu:
  omega-blocks: the LU-twirled moment E[(zz^*)^{(x)3}],
  gamma-blocks: the LU-twirled moment of w = z (x) z (x) eps^{(x)n} conj(z),
the four-qubit spin-flip construction.  Every block is written in an integer basis of
the standard S_3 representation (basis (1,-1,0),(0,1,-1)), so all kernels are
integers.  Every one-site antisymmetrizer of three qubit copies vanishes; the
resulting linear relations among orbit values are returned as `rels`.
"""
import itertools, math
from functools import reduce
import numpy as np

PERMS = [(0, 1, 2), (1, 0, 2), (2, 1, 0), (0, 2, 1), (1, 2, 0), (2, 0, 1)]  # e,(12),(13),(23),c,c^-1
def _comp(a, b): return tuple(a[b[i]] for i in range(3))
MUL = np.array([[PERMS.index(_comp(a, b)) for b in PERMS] for a in PERMS])
INV = np.array([PERMS.index(tuple(int(v) for v in np.argsort(p))) for p in PERMS])
SGN = np.array([1, -1, -1, -1, 1, 1])
_V = np.array([[1, 0], [-1, 1], [0, -1]])
def _pm(p):
    P = np.zeros((3, 3), int)
    for i, j in enumerate(p): P[j, i] = 1
    return P
RHO = []
for _p in PERMS:
    _R = np.linalg.lstsq(_V, _pm(_p) @ _V, rcond=None)[0]
    RHO.append(np.rint(_R).astype(np.int64))
GRAM = (_V.T @ _V).astype(np.int64)          # [[2,-1],[-1,2]]

def op8(p):
    """Permutation of the three tensor slots of (C^2)^{(x)3}."""
    M = np.zeros((8, 8), np.int64)
    for idx in itertools.product(range(2), repeat=3):
        out = [0]*3
        for i in range(3): out[p[i]] = idx[i]
        M[out[0]*4+out[1]*2+out[2], idx[0]*4+idx[1]*2+idx[2]] = 1
    return M
OP = [op8(p) for p in PERMS]
_EPS3 = np.kron(np.eye(4, dtype=np.int64), np.array([[0, 1], [-1, 0]]))
def _pt3(M): return M.reshape(2, 2, 2, 2, 2, 2).transpose(0, 1, 5, 3, 4, 2).reshape(8, 8)
def spin_flip_map(p):
    """t(pi) = (I(x)I(x)eps)^T pi^{T_3} (I(x)I(x)eps), as an 8x8 integer matrix."""
    return _EPS3.T @ _pt3(OP[p]) @ _EPS3
# 6*t(pi) = sum_sigma C6[pi,sigma] sigma  (integer expansion, checked in check_algebra)
C6 = np.array([[5, 1, 1, 1, -1, -1], [1, 5, -1, -1, 1, 1], [4, 2, -4, 2, -2, -2],
               [4, 2, 2, -4, -2, -2], [2, 4, -2, -2, -4, 2], [2, 4, -2, -2, 2, -4]], dtype=np.int64)

def check_algebra():
    for a in range(6):
        for b in range(6):
            c = MUL[a, b]
            assert (OP[a] @ OP[b] == OP[c]).all() and (RHO[a] @ RHO[b] == RHO[c]).all()
    assert sum(SGN[k]*OP[k] for k in range(6)).any() == False      # antisymmetrizer vanishes
    for a in range(6):
        assert (sum(C6[a, k]*OP[k] for k in range(6)) == 6*spin_flip_map(a)).all()
    return True

def orbits(n):
    W = np.array(list(itertools.product(range(6), repeat=n)), dtype=np.int64)
    pw = 6**np.arange(n)[::-1]
    best = None
    for g in range(6):
        for h in range(6):
            T = MUL[g][MUL[W, h]]
            for inv in (False, True):
                U = INV[T] if inv else T
                code = np.sort(U, axis=1) @ pw
                best = code if best is None else np.minimum(best, code)
    codes, orb = np.unique(best, return_inverse=True)
    return W, orb.reshape(-1), codes

def okron(ms, dtype=object):
    return reduce(np.kron, ms, np.array([[1]], dtype=dtype))

def _even_basis(m):
    ep = np.array([[1], [2]], dtype=object); em = np.array([[1], [0]], dtype=object)
    cols = [okron([em if s else ep for s in sg]) for sg in itertools.product((0, 1), repeat=m) if sum(sg) % 2 == 0]
    return np.hstack(cols) if cols else np.ones((1, 1), dtype=object)

def _s3_basis(m):
    import sympy as sp
    E = _even_basis(m)
    if m == 0: return E
    Cm = okron([np.array(RHO[4], dtype=object)]*m)
    cols = []
    for v in sp.Matrix((Cm.dot(E) - E).tolist()).nullspace():
        v = v*sp.ilcm(*[sp.fraction(x)[1] for x in v])
        iv = [int(x) for x in v]; g = reduce(math.gcd, iv); iv = [x//g for x in iv]
        cols.append(E.dot(np.array(iv, dtype=object)).reshape(-1, 1))
    return np.hstack(cols) if cols else None

def build(n):
    """Return (orbit codes, relations, blocks, o_norm, purity-word orbit function)."""
    W, orb, codes = orbits(n); NO = len(codes)
    pw = 6**np.arange(n)[::-1]
    def orbit_of(word): return int(orb[int(np.dot(word, pw))])
    rels = set()
    for rest in itertools.product(range(6), repeat=n-1):
        v = np.zeros(NO, dtype=np.int64)
        for s in range(6): v[orbit_of([s]+list(rest))] += SGN[s]
        if v.any():
            g = np.gcd.reduce(np.abs(v[v != 0])); v //= g
            if v[np.nonzero(v)[0][0]] < 0: v = -v
            rels.add(tuple(int(x) for x in v))
    rels = np.array(sorted(rels), dtype=np.int64)
    kin = {'omega': np.array([RHO[INV[s]] for s in range(6)]),
           'gamma': np.einsum('ps,pab->sab', C6, np.array([RHO[INV[p]] for p in range(6)]))}
    kout = {'omega': np.ones(6, dtype=np.int64), 'gamma': C6.sum(axis=0)}
    blocks = []
    for kind in ('omega', 'gamma'):
        for m in range(n+1):
            wt = np.prod(kout[kind][W[:, m:]], axis=1) if m < n else np.ones(len(W), dtype=np.int64)
            pre = W[:, :m] @ (6**np.arange(m)[::-1]) if m else np.zeros(len(W), dtype=np.int64)
            cnt = np.zeros((NO, 6**m), dtype=np.int64); np.add.at(cnt, (orb, pre), wt)
            V = _s3_basis(m) if kind == 'omega' else _even_basis(m)
            if V is None: continue
            U = okron([GRAM.astype(object)]*m).dot(V) if m else V
            Ks = []
            for o in range(NO):
                T = cnt[o].reshape((6,)*m) if m else cnt[o].reshape(())
                for _ in range(m): T = np.tensordot(T, kin[kind], axes=([0], [0]))
                if m:
                    T = T.transpose([2*i for i in range(m)]+[2*i+1 for i in range(m)]).reshape(2**m, 2**m)
                    T = okron([GRAM]*m, dtype=np.int64) @ T
                    Ks.append(U.T.dot(T.astype(object)).dot(U))
                else:
                    Ks.append(np.array([[int(T)]], dtype=object))
            Ks = np.array(Ks, dtype=object)
            g = reduce(math.gcd, [int(v) for v in Ks.reshape(-1)])
            if g == 0: continue
            blocks.append(dict(kind=kind, m=m, K=Ks//g, D=U.shape[1]))
    o_norm = orbit_of([0]*n)
    return dict(n=n, NO=NO, codes=codes, rels=rels, blocks=blocks, o_norm=o_norm, orbit_of=orbit_of, W=W, orb=orb)

def purity_orbit(data, k):
    n = data['n']; return data['orbit_of']([1]*k+[0]*(n-k))

def mu_word(psi, word):
    """<z^{(x)3}| (x)_i pi_i |z^{(x)3}> for a state tensor psi of shape (2,)*n."""
    n = psi.ndim; L = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    xs = [[L[3*i+c] for c in range(3)] for i in range(n)]
    ys = [[None]*3 for _ in range(n)]
    for i in range(n):
        for c in range(3): ys[i][PERMS[word[i]][c]] = xs[i][c]
    lhs = [''.join(xs[i][c] for i in range(n)) for c in range(3)]
    rhs = [''.join(ys[i][c] for i in range(n)) for c in range(3)]
    pc = psi.conj()
    return np.einsum(','.join(lhs+rhs)+'->', pc, pc, pc, psi, psi, psi, optimize='greedy')
