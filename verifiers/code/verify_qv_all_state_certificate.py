#!/usr/bin/env python3
"""QVA-1--QVA-4: exact all-state sharp quadratic purity-variance certificate (QV).

Proved in: quadratic_variance_certificate.tex and sup_renyi.tex#sec:sup-qk.
The certificate is a positive integer combination of 95 squared six-copy
covariant norms. Their sparse integer vectors have 716 nonzero entries,
all of absolute value at most 22. No positive-matrix test is needed.
Tensor contractions use checked int64 arithmetic; norm sums use Python integers.
Highest weights use [negative support, positive support]. Each covariant vector
is a flat [index, coefficient, ...] list with zero-based multiplicity indices.
The 83 determining states are shared with QG; their determining theorem is
an external input to this finite replay.
"""
import json
from itertools import permutations
from math import comb, lcm, prod
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

ORDERS = tuple(permutations(range(4)))
SIGNS = np.array([(-1)**b.bit_count() for b in range(16)], dtype=np.int64)
PERM3 = [4*c + q for q in range(4) for c in range(3)]
PERM_ONE = (1, 4, 2, 5, 0, 3, 6)
SWAPS = [(0, 1, 2, 3), (1, 0, 2, 3), (2, 1, 0, 3), (3, 1, 2, 0)]

HERE = Path(__file__).parent
DATA = json.loads((HERE / 'qv_all_state_certificate.json').read_text())
POINTS = json.loads((HERE / 'quadratic_purity_variance_certificate.json').read_text())['points']
assert DATA['format'] == 'qv-positive-norms-v1' and len(POINTS) == 83

# 1. Build and verify exact 6-copy SU(2) intertwiner C6 (64 x 64)
raising = np.zeros((64, 64), dtype=np.int64)
for b in range(64):
    for k in range(6):
        if (b >> k) & 1:
            raising[b ^ (1 << k), b] = 1

rows, SIZES, OFFSETS = [], {}, {}
for j in (0, 2, 4, 6):
    supports = DATA['highest_weight_supports'][str(j)]
    H = np.zeros((len(supports), 64), dtype=np.int64)
    for row, (negative, positive) in zip(H, supports):
        assert len(set(negative + positive)) == len(negative + positive)
        row[negative], row[positive] = -1, 1
    assert not (raising @ H.T).any()
    assert all(not H[:, b].any() for b in range(64) if b.bit_count() != (6 - j) // 2)
    SIZES[j] = (j + 1, len(H))
    OFFSETS[j] = sum(len(M) for M in rows)
    rows.append(H)
    for k in range(1, j + 1):
        H = H @ raising
        assert not (H % k).any()
        H = H // k
        rows.append(H)
C6 = np.concatenate(rows)
assert C6.shape == (64, 64)
C6_SPARSE = csr_matrix(C6)
C57_SPARSE = csr_matrix(C6[:57])  # a=4 uses only spins 0, 2, 4
C5_RV_U = np.ascontiguousarray(C6[:5].reshape(5, 8, 8).transpose(0, 2, 1).reshape(5 * 8, 8))
SLICE_MAP = {j: np.arange(OFFSETS[j], OFFSETS[j] + prod(SIZES[j])) for j in (0, 2, 4, 6)}


# Every block has p[0] == 0 (5 spin-0 rows). Swapping qubit order[0] to slot 0 keeps slot 0 at d0=5.
def make_grids(d):
    stride = 5 * (d ** 3)
    return [(q * stride + np.arange(stride, dtype=np.int32)).reshape(d, d, 5, d).transpose(2, 1, 0, 3).transpose(sw)
            for q, sw in enumerate(SWAPS)]


def extract_sector_swapped(grids, p):
    m_dim = prod(SIZES[j][0] for j in p)
    mult_dim = prod(SIZES[j][1] for j in p)
    res = []
    for order in ORDERS:
        spins = [0] * 4
        for k, i in enumerate(order):
            spins[i] = p[k]
        indices = [SLICE_MAP[j] for j in spins]
        c = grids[order[0]][np.ix_(*indices)].transpose(order)
        shape = sum(([SIZES[j][0], SIZES[j][1]] for j in p), [])
        c = c.reshape(shape).transpose(0, 2, 4, 6, 1, 3, 5, 7).reshape(m_dim, mult_dim)
        res.append(c)
    return np.vstack(res)


grids_64, grids_57 = make_grids(64), make_grids(57)
blks_by_a = {3: [], 4: []}
for idx_b, b in enumerate(DATA['blocks']):
    p = tuple(b['spins'])
    assert p[0] == 0
    wt = np.array([1], dtype=np.int64)
    for j in p:
        d_w = lcm(*(comb(j, m_i) for m_i in range(j + 1)))
        wt = np.kron(wt, np.array([d_w // comb(j, m_i) for m_i in range(j + 1)], dtype=np.int64))
    w_i64 = np.tile(wt, 24)
    vectors = b['vectors']
    assert len(vectors) == len(b['weights'])
    inputs = sorted({j for v in vectors for j in v[::2]})
    assert all(0 <= j < prod(SIZES[k][1] for k in p) for j in inputs)
    lookup = {j: i for i, j in enumerate(inputs)}
    rows, cols, vals = [], [], []
    for row, v in enumerate(vectors):
        assert len(v) % 2 == 0 and len(set(v[::2])) == len(v) // 2
        assert all(v[1::2])
        rows.extend([row] * (len(v) // 2))
        cols.extend(lookup[j] for j in v[::2])
        vals.extend(v[1::2])
    B = csr_matrix((vals, (rows, cols)), shape=(len(vectors), len(inputs)), dtype=np.int64)
    grids = grids_64 if b['a'] == 3 else grids_57
    indices = np.ascontiguousarray(extract_sector_swapped(grids, p)[:, inputs])
    blks_by_a[b['a']].append((idx_b, indices, B, w_i64))



def tensor3_flat(z0, z1, a3, omega=False):
    flip0 = (z0 - z1 if omega else z0)[np.arange(16) ^ 15] * SIGNS
    flip1 = -z1[np.arange(16) ^ 15] * SIGNS
    ua, ub = np.array([1], dtype=np.int64), np.array([0], dtype=np.int64)
    for k in range(3):
        u, v = (z0, z1) if k < a3 else (flip0, flip1)
        ac, bd = np.kron(ua, u), np.kron(ub, v)
        ua, ub = ac - bd, np.kron(ua, v) + np.kron(ub, u) - (bd if omega else 0)
    return (np.ascontiguousarray(ua.reshape([2]*12).transpose(PERM3).reshape(8, 8, 8, 8), dtype=np.int64),
            np.ascontiguousarray(ub.reshape([2]*12).transpose(PERM3).reshape(8, 8, 8, 8), dtype=np.int64))


def prep_u_swaps(ua, ub):
    return [(np.ascontiguousarray((C5_RV_U @ np.ascontiguousarray(ua.transpose(sw).reshape(8, 512))).reshape(5, 8, 512).transpose(0, 2, 1)).reshape(5 * 512, 8),
             np.ascontiguousarray((C5_RV_U @ np.ascontiguousarray(ub.transpose(sw).reshape(8, 512))).reshape(5, 8, 512).transpose(0, 2, 1)).reshape(5 * 512, 8))
            for sw in SWAPS]


def eval_swapped_pair(A_swaps, va, vb, C, omega=False):
    d = C.shape[0]
    out_a, out_b = [], []
    for (A_a, A_b), sw in zip(A_swaps, SWAPS):
        v_a = np.ascontiguousarray(va.transpose(sw).reshape(8, 512))
        v_b = np.ascontiguousarray(vb.transpose(sw).reshape(8, 512))
        ac, bd = A_a @ v_a, A_b @ v_b
        for w1, lst in ((ac - bd, out_a), (A_a @ v_b + A_b @ v_a - (bd if omega else 0), out_b)):
            w = np.ascontiguousarray(w1.reshape(5, 8, 8, 8, 8, 8, 8).transpose(PERM_ONE).reshape(64, 64, 5, 64))
            w = (C @ w.reshape(-1, 64).T).T.reshape(64, 64, 5 * d)
            w = (C @ w.reshape(64, -1)).reshape(d, 64, 5 * d)
            w_out = (C @ np.ascontiguousarray(w.transpose(1, 0, 2)).reshape(64, -1)).reshape(-1)
            lst.append(w_out)
    return np.concatenate(out_a), np.concatenate(out_b)


def evaluate(z0, z1, omega=False):
    # For these fixed samples, three-copy products have tiny exact entries.
    assert max(int(np.abs(z0).max()), int(np.abs(z1).max())) <= 2
    ua, ub = tensor3_flat(z0, z1, 3, omega)
    A = prep_u_swaps(ua, ub)
    for a, C in ((3, C6_SPARSE), (4, C57_SPARSE)):
        va, vb = tensor3_flat(z0, z1, a - 3, omega)
        # Absolute-sum bound covers every intermediate tensor contraction,
        # including multiplication in Z[omega], omega^2 = -1-omega.
        bound = 3 * max(int(np.abs(ua).max()), int(np.abs(ub).max()))
        bound *= max(int(np.abs(va).max()), int(np.abs(vb).max()))
        bound *= int(np.abs(C6).sum(axis=1).max()) ** 4
        assert bound < (1 << 60)
        wa, wb = eval_swapped_pair(A, va, vb, C, omega)
        for _, indices, B, weights in blks_by_a[a]:
            x, y = wa[indices], wb[indices]
            assert max(int(np.abs(x).max()), int(np.abs(y).max())) * int(abs(B).sum(axis=1).max()) < (1 << 60)
            yield B @ x.T, B @ y.T, weights


# QVA-1: only HS and cluster vanishing is needed for the determining space.
hs0, hs1 = np.zeros(16, dtype=np.int64), np.zeros(16, dtype=np.int64)
hs0[[3, 12]], hs1[[5, 10]] = 1, 1
hs0[[6, 9]], hs1[[6, 9]] = -1, -1
cluster = np.zeros(16, dtype=np.int64)
cluster[[0, 3, 12, 15]] = [1, 1, 1, -1]
for z0, z1, omega in ((hs0, hs1, True), (cluster, np.zeros(16, dtype=np.int64), False)):
    for x, y, _ in evaluate(z0, z1, omega):
        assert not x.any() and not y.any()
print('QVA-1 PASS: exact six-copy intertwiners; all 95 covariants vanish at HS and cluster')

# QVA-2: all scalar weights are positive integers.
D = int(DATA['denominator'])
assert D == 10235456181520
coefficients = [c for b in DATA['blocks'] for c in b['weights']]
assert len(DATA['blocks']) == 14 and len(coefficients) == 95 and min(coefficients) > 0
values = [c for b in DATA['blocks'] for v in b['vectors'] for c in v[1::2]]
assert len(values) == 716 and max(map(abs, values)) == 22
assert max(len(str(c)) for c in coefficients) == 14
print('QVA-2 PASS: 95 positive integer norm weights; denominator 10235456181520; at most 14 numerator digits')
print('Covariants: 716 nonzero integer coefficients, maximum absolute value 22')


def moments(z0, z1):
    # Object arrays make the marginal products arbitrary-precision integers.
    z0, z1 = z0.astype(object), z1.astype(object)
    n = sum(x*x + y*y for x, y in zip(z0, z1))
    ps, ts = [], []
    for j in (1, 2, 3):
        axes = [0, j] + [k for k in (1, 2, 3) if k != j]
        X, Y = [v.reshape([2]*4).transpose(axes).reshape(4, 4) for v in (z0, z1)]
        R, I = X @ X.T + Y @ Y.T, Y @ X.T - X @ Y.T
        ps.append(np.trace(R @ R - I @ I))
        ts.append(np.trace(R @ R @ R - 3 * R @ I @ I))
    return n, sum(ps), sum(ts), sum(p*p for p in ps)


def norm_features(pt):
    z0, z1 = np.array(pt, dtype=np.int64).T
    features = []
    for x, y, weights in evaluate(z0, z1):
        x, y = x.astype(object), y.astype(object)
        features.extend(int(v) for v in ((x*x + y*y) * weights).sum(axis=1))
    assert len(features) == len(coefficients)
    return features


# QVA-3: injectivity on the 83-dimensional invariant space is QG-1/QG-3.
for pt in POINTS:
    features = norm_features(pt)
    z0, z1 = np.array(pt, dtype=np.int64).T
    n, P, T, U = moments(z0, z1)
    F = 17*n*P - 12*T - 12*n**3
    twice_QV = 2*F*F + 10*n*n*P*P - 6*n**4*P - 12*n*n*U
    assert sum(x*c for x, c in zip(features, coefficients)) == D * twice_QV
print('QVA-3 PASS: exact identity 2*D*QV_defect = sum q_i*||s_i||^2 at all 83 determining states')

# QVA-4: retain the upstream regression of the equivariant construction.
def on_qubit(z0, z1, u0, u1, q):
    def act(u, z):
        return np.moveaxis(np.tensordot(u, np.moveaxis(z.reshape([2]*4), q, 0), axes=(1, 0)), 0, q).reshape(16)
    return np.column_stack((act(u0, z0) - act(u1, z1), act(u0, z1) + act(u1, z0))).tolist()


z0, z1 = np.array(POINTS[5], dtype=np.int64).T
f0 = norm_features(POINTS[5])
assert any(f0)
zero = np.zeros((2, 2), dtype=np.int64)
local_maps = [(np.array([[1, 1], [-1, 1]]), zero, 64),
              (np.eye(2, dtype=np.int64), np.array([[0, 1], [1, 0]]), 64),
              (np.diag([1, 0]), np.diag([0, 1]), 1),
              (np.array([[0, 1], [-1, 0]]), zero, 1)]
images = [(np.column_stack((z0, -z1)).tolist(), 1)]
for order in ((1, 0, 2, 3), (1, 2, 3, 0)):
    images.append((np.column_stack([z.reshape([2]*4).transpose(order).reshape(16) for z in (z0, z1)]).tolist(), 1))
images.extend((on_qubit(z0, z1, u0, u1, q), scale) for q in range(4) for u0, u1, scale in local_maps)
for pt, scale in images:
    assert norm_features(pt) == [scale * f for f in f0]
print('QVA-4 PASS: one-state regression: all 95 squared norms invariant under conjugation, qubit permutations and four local maps per qubit (with k^6 scaling)')
print('QVA PASS: positive-norm all-state QV certificate; integer arithmetic throughout')
