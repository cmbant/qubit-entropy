"""E7-K1: exact 17-block family images at six determining amplitude pairs.

Binary64 contractions are exact integer operations: every partial absolute
sum is bounded by 32**5*2**5*24**2*3**5 < 2**53. All subsequent projections
use integers. C5 placement orbits exhaust the site permutations.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "2"
import sys, time, itertools, math, json
from seven_entropy_data import load_bases
from pathlib import Path
import numpy as np, flint, opt_einsum as oe
from five_copy_physical_faces import (
    symmetric_gram_numerator,
    apply_gram,
    integer_matrix,
    models as fourmodels,
)
from five_copy_young51_model import sym_power
from five_copy_symmetric_models import symmetric_lookup, sym6_image
from certify_ico_fivecopy_g34_51 import (
    ico_integer_states,
    spinflip,
    local_maps,
    CYCLE,
    PATTERNS,
)
from certify_ico_fivecopy_allv_faces import row_sym, reps
from certify_ico_fivecopy_g34_42 import young42_image, pair_grouping
from five_copy_young42_model import model_and_projector
import verify_five_copy_site_algebra as s5

ROOT = Path(__file__).resolve().parent.parent / "data" / "seven_entropy"
D = load_bases()
CC = np.load(ROOT / "constraints.npz", allow_pickle=False)
defs = json.loads((ROOT / "models.json").read_text())
G4 = np.array(s5.V[1].T * s5.V[1], dtype=object)
pairs = list(itertools.combinations_with_replacement(range(4), 2))
wedges = list(itertools.combinations(range(4), 2))
Hsym = (
    np.array(
        [
            [
                G4[c, a] * G4[d, b] + (G4[d, a] * G4[c, b] if c != d else 0)
                for a, b in pairs
            ]
            for c, d in pairs
        ],
        object,
    )
    * np.array([2 if a == b else 1 for a, b in pairs], object)[:, None]
)
Halt = np.array(
    [[G4[c, a] * G4[d, b] - G4[d, a] * G4[c, b] for a, b in wedges] for c, d in wedges],
    object,
)
W = {}
start = time.time()
for i, M in enumerate(defs):
    U = D["u%d" % i]
    # H fixedness suffices; completeness is not assumed for these SOS sources.
    gen = [
        j
        for j in range(4)
        if M["flip"] == 0
        or j < (4 if M["flip"] == 1 else 3) - 1
        or (M["flip"] == 2 and j == 3)
    ]
    for j in gen:
        p = list(range(5))
        p[j], p[j + 1] = p[j + 1], p[j]
        fac = []
        for d, n in M["fac"]:
            R = np.array(s5.RHO[1 if d == 4 else 2][s5.PINDEX[tuple(p)]], np.int64)
            if d == 4 and n == 2:
                pp = wedges if M["wedge"] else pairs
                A = np.array(
                    [
                        [
                            R[c, a] * R[d, b]
                            + (
                                -R[d, a] * R[c, b]
                                if M["wedge"]
                                else (R[d, a] * R[c, b] if c != d else 0)
                            )
                            for a, b in pp
                        ]
                        for c, d in pp
                    ],
                    np.int64,
                )
            else:
                A = R if n == 1 else sym_power(R, n)
            fac.append(A)
        X = U.reshape(*M["dims"], U.shape[1])
        for j, A in enumerate(fac):
            assert A.shape[1] * int(abs(A).max()) * int(abs(X).max()) < 2**63
            X = np.moveaxis(np.tensordot(A, X, axes=([1], [j])), 0, j)
        assert np.array_equal(X.reshape(U.shape), U)
    if i in [1, 2, 3]:
        W[i] = integer_matrix(U).transpose()
        continue
    fs = []
    for d, n in M["fac"]:
        fs.append(
            (Halt if M["wedge"] else Hsym)
            if d == 4 and n == 2
            else G4 if d == 4 else symmetric_gram_numerator(n)
        )
    W[i] = integer_matrix(apply_gram(U, fs)).transpose()
print("EXACT H FIXEDNESS PASS", 17, flush=True)
for i, (_, U, _, _, _) in enumerate(fourmodels()):
    assert np.array_equal(U, D["u%d" % i])
_, P40, _ = model_and_projector()
sort42, start42 = pair_grouping()
placements = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4)]
seen = set()
for pp in placements:
    for k in range(5):
        seen.add(pp)
        pp = tuple(sorted(CYCLE[j] for j in pp))
assert len(seen) == 21
maps = local_maps()
u, v = ico_integer_states()
F = {
    m: maps[m]
    .astype(float)
    .reshape((maps[m].shape[0],) + (2,) * 5)
    .transpose((1, 2, 3, 4, 5, 0))
    for m in [1, 2]
}
paths = {}
o5, st5 = symmetric_lookup(5)
grp = np.array(
    [pairs.index(tuple(sorted(p))) for p in itertools.product(range(4), repeat=2)]
)
o4 = np.argsort(grp, kind="stable")
st4 = np.r_[0, np.where(np.diff(grp[o4]))[0] + 1]


def check(i, a, b, X):
    # Expand one original image at a time, with int64 arithmetic.
    columns = CC["c%d" % i].astype(np.int64)
    index = CC["i%d" % i].astype(np.int64)
    assert np.all((index >= 0) & (index < columns.shape[2]))
    C = columns[:, :, index]
    Y = sum(a ** (5 - k) * b**k * C[k].astype(object) for k in range(6))
    assert X == integer_matrix(Y), (i, a, b)
    print("EXACT IMAGE PASS", i, a, b, "sec", time.time() - start, flush=True)


def contract(z, jz, pattern, flip):
    key = tuple(pattern)
    ops = []
    for cc in range(5):
        ops += [z if cc < 5 - flip else jz, [7 * cc + i for i in range(7)]]
    for i, m in enumerate(pattern):
        ops += [F[m], [7 * cc + i for cc in range(5)] + [35 + i]]
    if key not in paths:
        paths[key] = oe.contract_path(
            *ops, list(range(35, 42)), optimize="dp", memory_limit=2**29
        )[0]
    assert 32**5 * 2**5 * 24 ** pattern.count(1) * 3 ** pattern.count(2) < 2**53
    T = oe.contract(*ops, list(range(35, 42)), optimize=paths[key])
    assert np.array_equal(T, np.rint(T))
    return T.astype(np.int64)


def y51(T, pat):
    order = [pat.index(1)] + [i for i in range(7) if i != pat.index(1)]
    X = (
        T.transpose(order)
        .reshape(4, 4, *([5, 2] * 6))
        .transpose(
            [0] + [2 + 2 * j for j in range(6)] + [1] + [3 + 2 * j for j in range(6)]
        )
        .reshape(4, *([5] * 6), 256)
    )
    oo, ss = symmetric_lookup(5)
    YY = []
    for j in range(6):
        P = X.transpose([0] + [1 + i for i in range(6) if i != j] + [1 + j, 7]).reshape(
            4, 3125, 5, 256
        )
        YY.append(np.add.reduceat(P[:, oo], ss, axis=1).reshape(2520, 256))
    assert 12 * 120 * int(abs(T).max()) < 2**63
    total = sum(YY)
    return np.hstack([6 * Y - total for Y in YY[:5]])


for a, b in [(1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (2, 1)]:
    z = a * u + b * v
    jz = spinflip(z)
    XX = {12: [], 13: []}
    for pp in placements:
        T = contract(z, jz, [1 if i in pp else 2 for i in range(7)], 2)
        order = list(pp) + [i for i in range(7) if i not in pp]
        dims = [(4, 4) if i in pp else (5, 2) for i in range(7)]
        T = (
            T.reshape(*[d for dd in dims for d in dd])
            .transpose([2 * i for i in order] + [2 * i + 1 for i in order])
            .reshape(16, 3125, 512)
        )
        Tv = np.add.reduceat(T[:, o5], st5, axis=1)
        Ys = np.add.reduceat(Tv[o4], st4, axis=0).reshape(1260, 512)
        Ya = np.array([Tv[4 * c + d] - Tv[4 * d + c] for c, d in wedges]).reshape(
            756, 512
        )
        XX[12].append(W[12] * integer_matrix(Ys))
        XX[13].append(W[13] * integer_matrix(Ya))
        del T, Tv, Ys, Ya
    for i in [12, 13]:
        vals = np.hstack(
            [
                np.array([int(w) for w in X.entries()], object).reshape(
                    X.nrows(), X.ncols()
                )
                for X in XX[i]
            ]
        )
        check(i, a, b, integer_matrix(vals))
    for flip in [0, 1, 2]:
        T = (
            contract(z, jz, [2] * 7, flip)
            .reshape(*([5, 2] * 7))
            .transpose([2 * i for i in range(7)] + [2 * i + 1 for i in range(7)])
        )
        for i, M in enumerate(defs):
            if M["flip"] != flip or i in [1, 2, 3, 12, 13]:
                continue
            k = M["fac"][0][1]
            l = M["fac"][1][1] if len(M["fac"]) > 1 else 0
            Y = np.hstack([row_sym(T, k, l, s) for s in reps(l)])
            check(i, a, b, W[i] * integer_matrix(Y))
        del T, Y
    XX = {i: [] for i in [1, 2, 3]}
    for pat in PATTERNS:
        T = contract(z, jz, list(pat), 2)
        for i, Y in [
            (1, sym6_image(T, pat, *symmetric_lookup(6))),
            (2, y51(T, pat)),
            (3, young42_image(T, pat, P40, sort42, start42)),
        ]:
            XX[i].append(W[i] * integer_matrix(Y))
        del T, Y
    for i in [1, 2, 3]:
        vals = np.hstack(
            [
                np.array([int(w) for w in X.entries()], object).reshape(
                    X.nrows(), X.ncols()
                )
                for X in XX[i]
            ]
        )
        check(i, a, b, integer_matrix(vals))
print(
    "E7-K1 PASS: exact physical degree-five images for all 17 blocks",
    time.time() - start,
    flush=True,
)
