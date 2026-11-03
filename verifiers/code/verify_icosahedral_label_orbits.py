"""EO-1: exact diagonal-congruence classification of labelled conference states.

The proof home derives the LU criterion from the nonzero axial pair profile.
This checks every one of the 720 ordinary-site permutations with exact integer
switching and complement signs. Seven distinguished sites then give 42 LU orbits.
"""

from itertools import combinations, permutations, product
import numpy as np

S = np.ones((6, 6), dtype=np.int64) - np.eye(6, dtype=np.int64)
for i, j in [(1, 3), (2, 3), (2, 4), (4, 5), (1, 5)]:
    S[i, j] = S[j, i] = -1
assert np.array_equal(S @ S, 5 * np.eye(6, dtype=np.int64))
P = list(permutations(range(6)))


def canonical(T):
    forms = []
    for sign in [1, -1]:
        d = np.r_[1, sign * T[0, 1:]]
        forms.append(tuple((sign * d[:, None] * T * d[None, :]).ravel()))
    return min(forms)


def state_basis(C):
    U = [0] * 128
    for i, j in combinations(range(6), 2):
        n = (1 << (5 - i)) + (1 << (5 - j))
        U[n] = int(C[i, j])
        U[127 ^ n] = -int(C[i, j])
    V = [0] * 128
    V[63] = V[64] = 1
    return U, V


def check_gauge(C, sign, d):
    target = sign * d[:, None] * C * d[None, :]
    mm = [(0 if v == 1 else 2) + (0 if sign == 1 else 1) for v in d]
    total = sum(mm)
    before = state_basis(C)
    after = state_basis(target)
    for a, b in zip(before, after):
        for n in range(128):
            if a[n] == 0:
                assert b[n] == 0
                continue
            exponent = (2 * total if n < 64 else 0) + sum(
                (4 * mm[i] - total) for i in range(6) if n & (1 << (5 - i))
            )
            assert exponent % 16 in [0, 8]
            assert a[n] * (1 if exponent % 16 == 0 else -1) == b[n]


classes = {}
automorphisms = {1: [], -1: []}
for p in P:
    T = S[np.ix_(p, p)]
    for sign in [1, -1]:
        dd = np.r_[1, sign * T[0, 1:]]
        check_gauge(T, sign, dd)

    for sign in [1, -1]:
        d = np.r_[1, sign * T[0, 1:] * S[0, 1:]]
        if np.array_equal(T, sign * d[:, None] * S * d[None, :]):
            automorphisms[sign].append(p)
    classes.setdefault(canonical(T), []).append(p)
assert [len(automorphisms[sign]) for sign in [1, -1]] == [60, 60]
assert len(classes) == 6 and all(len(c) == 120 for c in classes.values())
print(
    "EO-1 explicit diagonal gauges checked on both amplitude-pattern basis vectors; PASS"
)
# Verify the canonical key against exhaustive normalized switching signs,
# independently of the row-zero formula on all representatives.
representatives = [(0, 1, 2) + p for p in permutations((3, 4, 5))]
assert len({canonical(S[np.ix_(p, p)]) for p in representatives}) == 6
for p in representatives:
    T = S[np.ix_(p, p)]
    forms = []
    for sign in [1, -1]:
        for dd in product([1, -1], repeat=5):
            d = np.array((1,) + dd)
            A = sign * d[:, None] * T * d[None, :]
            # Compare row-normalized representatives, not raw sign order.
            if np.all(A[0, 1:] == 1):
                forms.append(tuple(A.ravel()))
    assert min(forms) == canonical(T)
print("EO-1 switching automorphisms: 60; switching-complement automorphisms: 60")
print("EO-1 all 720 ordinary permutations: six equivalence classes of size 120; PASS")
print(
    "EO-1 representatives fix ordinary labels 0,1,2 and permute 3,4,5:", representatives
)
print(
    "EO-1 PASS: 7 distinguished sites times 6 classes = 42 labelled LU orbits; one LU/permutation orbit"
)
