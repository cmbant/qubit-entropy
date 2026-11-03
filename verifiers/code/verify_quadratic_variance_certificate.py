#!/usr/bin/env python3
"""QG-1--QG-4: exact sparse covariant-norm certificate for universal QK.

Proved in: quadratic_variance_certificate.tex and sup_renyi.tex#sec:sup-qk.
The supplied integer directions define 80 nonnegative squared norms. Their
positive rational coefficients are computed by an exact 80x80 solve. An exact
83-point determining matrix proves both polynomial identities universally.
The kernel dimension 83 is the SD-4 dimension 85 minus the two HS/cluster
evaluations, whose moment values are computed here from the cut spectra.
Integer and exact rational arithmetic only.
"""

from collections import defaultdict
from pathlib import Path
import json
import time
import numpy as np
import sympy as sp
from flint import fmpq_mat, fmpz_mat
from quadratic_covariants import QuadraticCovariants

DATA_PATH = Path(__file__).with_name("quadratic_purity_variance_certificate.json")
DATA = json.loads(DATA_PATH.read_text())
C = QuadraticCovariants(DATA)
RAYS = defaultdict(list)
for k, ray in enumerate(DATA["rays"]):
    block = DATA["blocks"][ray["block"]]
    v = fmpz_mat([[int(t)] for t in ray["vector"]])
    assert len(ray["vector"]) == block["shape"][1]
    assert any(ray["vector"])
    integer_vector = C.transforms[block["raw"]] * v
    vec = [int(integer_vector[i, 0]) for i in range(integer_vector.nrows())]
    assert max(map(abs, vec)) < 2**40
    RAYS[block["raw"]].append((k, np.array(vec, dtype=np.int64)))
assert len(DATA["rays"]) == 80 and len(RAYS) == 16


def moments(z0, z1):
    n = sum(int(x) ** 2 + int(y) ** 2 for x, y in zip(z0, z1))
    ps, ts = [], []
    for j in (1, 2, 3):
        axes = [0, j] + [k for k in (1, 2, 3) if k != j]
        X, Y = [v.reshape([2] * 4).transpose(axes).reshape(4, 4) for v in (z0, z1)]
        R, I = X @ X.T + Y @ Y.T, Y @ X.T - X @ Y.T
        ps.append(int(np.trace(R @ R - I @ I)))
        ts.append(int(np.trace(R @ R @ R - 3 * R @ I @ I)))
    return n, sum(ps), sum(ts), sum(p * p for p in ps)


def basis_positions(components):
    wanted = set(DATA["basis_columns"])
    out, pos = {}, 0
    for raw, (X, _) in enumerate(components):
        for i in range(X.shape[-1]):
            for j in range(i, X.shape[-1]):
                if pos in wanted:
                    out[pos] = (raw, i, j)
                pos += 1
    assert len(out) == len(wanted) == 83
    return [out[k] for k in DATA["basis_columns"]]


def features(z0, z1, need_basis=True):
    cs, weights = C.components(z0, z1)
    n = sum(int(x) ** 2 + int(y) ** 2 for x, y in zip(z0, z1))
    norms = [0] * 80
    flat = []
    for raw, ((X, Y), wt) in enumerate(zip(cs, weights)):
        X, Y = [v.reshape(-1, v.shape[-1]) for v in (X, Y)]
        wt = np.tile(wt, 24)
        factor = n if raw < len(C.others) else 1
        assert all(int(w) > 0 for w in wt)
        flat.append((X, Y, wt, factor))
        for index, vec in RAYS.get(raw, []):
            # Every int64 dot product is exact under this absolute-sum bound.
            bound = max(int(np.abs(X).max()), int(np.abs(Y).max()))
            assert bound * sum(abs(int(v)) for v in vec) < 2**62
            xx, yy = X @ vec, Y @ vec
            norms[index] = factor * sum(
                int(w) * (int(a) ** 2 + int(b) ** 2) for w, a, b in zip(wt, xx, yy)
            )
    if not need_basis:
        return norms
    base = []
    for raw, i, j in basis_positions(cs):
        X, Y, wt, factor = flat[raw]
        value = sum(
            int(w) * (int(a) * int(b) + int(c) * int(d))
            for w, a, b, c, d in zip(wt, X[:, i], X[:, j], Y[:, i], Y[:, j])
        )
        base.append(factor * value * (1 if i == j else 2))
    return norms, base


def targets(n, P, T, U):
    F = 17 * n * P - 12 * T - 12 * n**3
    qk = 2 * F * F + 9 * n**4 * (P - n * n) - 12 * n * n * U + 4 * n * n * P * P
    return qk, n * n * (P - n * n) * (3 * n * n - 2 * P)


def spectral_moments(cuts):
    ps = [sum(x**2 for x in c) for c in cuts]
    return 1, sum(ps), sum(x**3 for c in cuts for x in c), sum(p * p for p in ps)


def main():
    start = time.monotonic()
    C.check_zeros()
    R = sp.Rational
    hs = spectral_moments([[R(1, 2)] + [R(1, 6)] * 3] * 3)
    cluster = spectral_moments([[R(1, 2), R(1, 2), 0, 0], [R(1, 4)] * 4, [R(1, 4)] * 4])
    assert (hs[2], cluster[2]) == (R(5, 12), R(3, 8))
    assert targets(*hs) == targets(*cluster) == (0, 0)
    assert sp.Matrix([[1, hs[2]], [1, cluster[2]]]).det() != 0
    print(
        "QG-1 PASS: integer spin intertwiners; raw entries and both targets vanish at HS and cluster;"
        " independent (N^6, N^3T) evaluations (1,5/12), (1,3/8); kernel dimension 85 (SD-4) - 2 = 83",
        flush=True,
    )
    A, basis, target, zero_target = [], [], [], []
    for i, point in enumerate(DATA["points"]):
        z0, z1 = np.array(point, dtype=np.int64).T
        vals, base = features(z0, z1)
        n, P, T, U = moments(z0, z1)
        A.append(vals)
        basis.append(base)
        qk, zero = targets(n, P, T, U)
        target.append(qk)
        zero_target.append(zero)
        if (i + 1) % 10 == 0:
            print(f"QG exact evaluations: {i+1}/83", flush=True)
    prime = DATA["prime"]
    assert sp.isprime(prime) and len(basis) == len(basis[0]) == 83
    assert int(fmpz_mat(basis).det()) % prime == DATA["det_mod_prime"] == 627608
    rows = DATA["solve_rows"]
    assert len(rows) == len(set(rows)) == 80 and all(0 <= i < 83 for i in rows)
    E = fmpq_mat([A[i] for i in rows])
    rhs = fmpq_mat([[target[i], zero_target[i]] for i in rows])
    solution = E.solve(rhs)
    assert all(solution[i, 0] > 0 for i in range(80))
    print(
        "QG-2 PASS: all 80 scalar coefficients strictly positive by an exact 80x80 solve; 16 covariant modules",
        flush=True,
    )
    full = fmpq_mat(A) * solution
    assert all(full[i, 0] == target[i] for i in range(83))
    assert all(full[i, 1] == zero_target[i] for i in range(83))
    print(
        "QG-3 PASS: full sparse norm identity; determining determinant = 627608 mod 1000003",
        flush=True,
    )
    print(
        "QG-4 PASS: signed norm-span identity n^2(P-n^2)(3n^2-2P); every zero has P=1 or 3/2",
        flush=True,
    )
    print(f"Runtime: {time.monotonic()-start:.2f} s", flush=True)


if __name__ == "__main__":
    main()
