"""E7-K0: exact [4,3] membership and independence of the added invariant bases."""

from seven_entropy_data import load_bases, load_constraints
from pathlib import Path
import math
import numpy as np
import scipy.sparse as sparse
import flint
from five_copy_probe_models import mons
import verify_five_copy_site_algebra as s5

ROOT = Path(__file__).resolve().parent.parent / "data" / "seven_entropy"
D = load_bases()
m4, m3, m5, m2 = [mons(5, n) for n in [4, 3, 5, 2]]
pos5, pos2 = [{m: i for i, m in enumerate(mm)} for mm in [m5, m2]]
rows, cols, values = [], [], []
for i, a in enumerate(m4):
    for j, b in enumerate(m3):
        for t in range(5):
            if b[t]:
                aa, bb = list(a), list(b)
                aa[t] += 1
                bb[t] -= 1
                rows.append(pos5[tuple(aa)] * len(m2) + pos2[tuple(bb)])
                cols.append(i * len(m3) + j)
                values.append(b[t])
raising = sparse.csr_matrix((values, (rows, cols)), shape=(1890, 2450), dtype=np.int64)


def symmetric_characters(R, n):
    R = np.array(R, dtype=object)
    A = np.eye(len(R), dtype=object)
    powers = [0]
    for j in range(1, n + 1):
        A = A @ R
        powers.append(sum(int(A[k, k]) for k in range(len(R))))
    h = [1]
    for j in range(1, n + 1):
        numerator = sum(powers[k] * h[j - k] for k in range(1, j + 1))
        assert numerator % j == 0
        h.append(numerator // j)
    return h


for i, n, flip in [(12, 116, 2), (13, 60, 2), (14, 4, 0), (15, 22, 1), (16, 46, 2)]:
    H = [
        k
        for k, p in enumerate(s5.PERMS)
        if flip == 0 or (p[4] == 4 if flip == 1 else set(p[3:]) == {3, 4})
    ]
    total = 0
    for k in H:
        h = symmetric_characters(s5.RHO[2][k], 5)
        if i in [12, 13]:
            R = np.array(s5.RHO[1][k], dtype=object)
            t = sum(int(R[j, j]) for j in range(4))
            R2 = R @ R
            t2 = sum(int(R2[j, j]) for j in range(4))
            numerator = t * t + (t2 if i == 12 else -t2)
            assert numerator % 2 == 0
            total += numerator // 2 * h[5]
        else:
            total += h[4] * h[3] - h[5] * h[2]
    assert total == len(H) * n
    U = D["u%d" % i]
    assert U.shape[1] == n
    if i >= 14:
        assert U.shape[0] == 2450
        assert int(abs(raising).sum(axis=1).max()) * int(abs(U).max()) < 2**63
        assert not np.any(raising @ U)
    Q = flint.fmpz_mat(U.shape[0], U.shape[1], [int(v) for v in U.ravel()])
    assert (Q.transpose() * Q).det() != 0
    print(
        "E7-K0 exact invariant character dimension and independent columns",
        i,
        n,
        "PASS",
    )
print("E7-K0 PASS: complete added invariant bases, with exact Young [4,3] membership")
