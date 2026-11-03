#!/usr/bin/env python3
"""ES-1: exact fixed-space algebra for symmetric criticality of the conference family.
The proof for every differentiable invariant functional is analytic in
seven_qubit_entropy_local.tex. No marginal-power contractions or floats are used.
"""

import itertools
import numpy as np

N = 7
CUTS = [
    (tuple(c), tuple(i for i in range(N) if i not in c))
    for c in itertools.combinations(range(N), 3)
]
PENTAGON = {(1, 3), (2, 3), (2, 4), (4, 5), (1, 5)}
U = np.zeros((2,) * N, dtype=np.int64)
V = np.zeros_like(U)
for i, j in itertools.combinations(range(6), 2):
    sg = -1 if (i, j) in PENTAGON else 1
    p = [0] * 7
    q = [1] * 7
    p[1 + i] = p[1 + j] = 1
    q[1 + i] = q[1 + j] = 0
    U[tuple(p)] = sg
    U[tuple(q)] = -sg
V[(0, 1, 1, 1, 1, 1, 1)] = V[(1, 0, 0, 0, 0, 0, 0)] = 1
assert (np.sum(U * U), np.sum(V * V), np.sum(U * V)) == (30, 2, 0)


def symmetry_checks():
    from sympy import Matrix

    C = np.ones((6, 6), dtype=np.int64) - np.eye(6, dtype=np.int64)
    for i, j in PENTAGON:
        C[i, j] = C[j, i] = -1
    g = [0, 3, 4, 2, 5, 1]  # cycle (1 3 2 4 5)
    p = [1, 0, 4, 3, 2, 5]  # (0 1)(2 4)
    eps = np.array([1, 1, 1, -1, 1, -1], dtype=np.int64)
    assert np.array_equal(C, C[np.ix_(g, g)])
    assert np.array_equal(C, C[np.ix_(p, p)] * np.outer(eps, eps))
    bits = [tuple((j >> (6 - i)) & 1 for i in range(7)) for j in range(128)]
    index = {b: j for j, b in enumerate(bits)}
    support = [j for j, b in enumerate(bits) if (2 + 2 * b[0] + sum(b[1:])) % 4 == 0]
    assert len(support) == 32
    where = {j: i for i, j in enumerate(support)}

    def permute(b, pi):
        out = [b[0]] + [0] * 6
        for i in range(6):
            out[pi[i] + 1] = b[i + 1]
        return tuple(out)

    Pg = np.zeros((32, 32), dtype=np.int64)
    W = Pg.copy()
    Q = Pg.copy()
    for col, j in enumerate(support):
        b = bits[j]
        Pg[where[index[permute(b, g)]], col] = 1
        target = permute(b, p)
        exponent = (
            2 * (1 - target[0])
            + sum((3 if eps[i] == 1 else 1) * target[i + 1] for i in range(6))
        ) % 4
        assert exponent in (0, 2)
        W[where[index[target]], col] = 1 if exponent == 0 else -1
        target = tuple(1 - v for v in b)
        # The eighth-root phase is real on this support; avoid floats entirely.
        exponent = (2 * (1 - target[0]) + sum(target[1:])) % 8
        assert exponent in (0, 4)
        Q[where[index[target]], col] = 1 if exponent == 0 else -1
    eye = np.eye(32, dtype=np.int64)
    uv = [t.reshape(128)[support] for t in (U, V)]
    for action in (Pg, W, Q):
        assert np.array_equal(action.T @ action, eye)
        assert all(np.array_equal(action @ v, v) for v in uv)
    before = Matrix(np.vstack([Pg - eye, W - eye]).tolist()).rank()
    after = Matrix(np.vstack([Pg - eye, W - eye, Q - eye]).tolist()).rank()
    assert before == 28 and after == 30
    assert all(
        not t.reshape(128)[[j for j in range(128) if j not in where]].any()
        for t in (U, V)
    )
    print("PASS ES-1: exact phase support and two conference switching symmetries")
    print(
        "PASS ES-1: common real fixed space 4 -> 2 after the local complement symmetry; basis U,V"
    )
    print(
        "Analytic conclusion: every differentiable LU/permutation/conjugation-invariant functional"
    )
    print(
        "is fully stationary at each interior stationary point of its conference-family restriction."
    )


if __name__ == "__main__":
    symmetry_checks()
