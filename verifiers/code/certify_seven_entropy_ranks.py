"""E7-K2: exact polynomial rank bounds for all 17 physical images."""

import os, time, json

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np, flint
from seven_entropy_data import load_bases, load_constraints
from pathlib import Path

root = str(Path(__file__).resolve().parent.parent / "data" / "seven_entropy")
D = load_constraints()
defs = json.load(open(root + "/models.json"))
start = time.time()
p = 2147483629
choose = {}


def fm(A):
    return flint.fmpz_mat(A.shape[0], A.shape[1], [int(x) for x in A.ravel()])


def pivots(M):
    F = flint.nmod_mat(M.nrows(), M.ncols(), [int(v) % p for v in M.entries()], p)
    F, r = F.rref()
    J = [next(j for j in range(M.ncols()) if F[i, j]) for i in range(r)]
    F = flint.nmod_mat(
        r, M.nrows(), [int(M[i, j]) % p for j in J for i in range(M.nrows())], p
    )
    F, rr = F.rref()
    assert rr == r
    I = [next(j for j in range(M.nrows()) if F[i, j]) for i in range(r)]
    return I, J


for idx, d in enumerate(defs):
    C = [fm(c) for c in D["c%d" % idx]]
    r = d["rank"]
    num = 5 * (r + 1) + 1

    def ev(t):
        M = C[5]
        for j in range(4, -1, -1):
            M = t * M + C[j]
        return M

    I, J = pivots(ev(1))
    assert len(I) == r
    choose[str(idx)] = dict(rows=I, columns=J)
    for t in range(num):
        M = ev(t)
        B = flint.fmpz_mat(r, r, [M[i, j] for i in I for j in J])
        dd = B.det()
        II, JJ = I, J
        if not dd:
            II, JJ = pivots(M)
            assert len(II) <= r
            B = flint.fmpz_mat(len(II), len(JJ), [M[i, j] for i in II for j in JJ])
            dd = B.det()
        if not II:
            assert M.is_zero()
            continue
        assert dd
        inv = flint.fmpq_mat(B).inv() * dd
        assert all(v.denom() == 1 for v in inv.entries())
        adj = flint.fmpz_mat(len(II), len(JJ), [int(v) for v in inv.entries()])
        L = flint.fmpz_mat(
            M.nrows(), len(JJ), [M[i, j] for i in range(M.nrows()) for j in JJ]
        )
        R = flint.fmpz_mat(
            len(II), M.ncols(), [M[i, j] for i in II for j in range(M.ncols())]
        )
        assert (dd * M - L * adj * R).is_zero()
    print(
        "GENERIC EXACT RANK UPPER",
        idx,
        r,
        "points",
        num,
        "sec",
        time.time() - start,
        flush=True,
    )
assert choose == json.load(open(root + "/pivots.json"))
print("ALL 17 GENERIC RANK BOUNDS PASS", time.time() - start, flush=True)
