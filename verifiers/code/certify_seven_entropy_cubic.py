"""E7-K3: exact cubic physical images, generating matrices and local trace maps."""

import os, sys, itertools, math, time, json

os.environ["OPENBLAS_NUM_THREADS"] = "1"
from pathlib import Path
import numpy as np, sympy as sp, flint, opt_einsum as oe
from five_copy_probe_models import mons
from five_copy_young51_model import select_columns
from certify_ico_fivecopy_g34_51 import ico_integer_states, spinflip
from five_copy_probe_evaluation import cosets
import verify_five_copy_site_algebra as s5

root = str(Path(__file__).resolve().parent.parent / "data" / "seven_entropy")
start = time.time()
PP = list(itertools.permutations(range(3)))


def op(p):
    O = np.zeros((8, 8), int)
    for x in itertools.product(range(2), repeat=3):
        y = [0] * 3
        for i in range(3):
            y[p[i]] = x[i]
        O[int("".join(map(str, y)), 2), int("".join(map(str, x)), 2)] = 1
    return O


OPS = [op(p) for p in PP]
V = sp.Matrix([[-1, -1], [1, 0], [0, 1]])
VV = sp.zeros(8, 2)
for i, j in enumerate([4, 2, 1]):
    VV[j, :] = V[i, :]
V0 = sp.eye(8)[:, 0]
L = (VV.T * VV).inv() * VV.T
R = [L * sp.Matrix(O) * VV for O in OPS]
G = V.T * V
H = [i for i, p in enumerate(PP) if p[2] == 2]


def sy(A, n):
    # monomial order (n,0),(n-1,1),...,(0,n)
    out = []
    for r in range(n + 1):
        row = []
        for c in range(n + 1):
            val = 0
            for k in range(max(0, r - c), min(n - c, r) + 1):
                val += (
                    math.comb(n - c, k)
                    * math.comb(c, r - k)
                    * A[0, 0] ** (n - c - k)
                    * A[1, 0] ** k
                    * A[0, 1] ** (c - r + k)
                    * A[1, 1] ** (r - k)
                )
            row.append(val)
        out.append(row)
    return np.array(out, object)


def fmat(A):
    return flint.fmpq_mat(
        [[flint.fmpq(str(v)) for v in row] for row in np.array(A, object).tolist()]
    )


def gram(n):
    return (
        np.array(
            [math.factorial(n - r) * math.factorial(r) for r in range(n + 1)], object
        )[:, None]
        * sy(np.array(G, object), n)
        / math.factorial(n)
    )


# exact invariant bases, including Young[6,1]=kernel(Sym6 V tensor V ->Sym7 V)
Mul = np.zeros((8, 14), int)
for c in range(7):
    Mul[c, 2 * c] = 1
    Mul[c + 1, 2 * c + 1] = 1
B = np.array(sp.Matrix.hstack(*sp.Matrix(Mul).nullspace()), int)
actions = [
    [sy(np.array(R[i], object), 7) for i in H],
    [np.kron(sy(np.array(R[i], object), 6), np.array(R[i], object)) for i in H],
    [sy(np.array(R[i], object), 6) for i in H],
]
bases = [np.eye(8, dtype=int), B, np.eye(7, dtype=int)]
Us = [
    select_columns(np.array(sum(A @ B for A in As), dtype=np.int64))
    for As, B in zip(actions, bases)
]
assert [U.shape[1] for U in Us] == [4, 3, 4]
Gs = [gram(7), np.kron(gram(6), np.array(G, object)), gram(6)]
Gs = [np.array([[sp.Rational(v) for v in row] for row in GG], object) for GG in Gs]
# exact local physical coordinate maps for spin3/2 and spin1/2
J = sp.zeros(8)
for x in range(8):
    for c in range(3):
        mask = 1 << (2 - c)
        if not x & mask:
            J[x | mask, x] += 1
Fs = []
for Q, mult in [(V0, 4), (VV, 2)]:
    desc = []
    X = Q
    for k in range(mult):
        desc.append(X)
        X = J * X
    S = sp.Matrix.hstack(*[desc[k][:, i] for i in range(Q.cols) for k in range(mult)])
    F = (S.T * S).inv() * S.T
    den = sp.ilcm(*[v.q for v in F])
    Fs.append(
        np.array(F * den, int)
        .reshape(Q.cols * mult, 2, 2, 2)
        .transpose(1, 2, 3, 0)
        .astype(float)
    )
assert 32**3 * 3**3 * 6**7 < 2**53
u, v = ico_integer_states()
images = [[], [], []]
paths = {}
for t in range(4):
    z = u + t * v
    jz = spinflip(z)
    YY = [[], [], []]
    for fpos in [None, 0, 1, 2]:
        ops = []
        for cc in range(3):
            ops += [z if cc < 2 else jz, [7 * cc + i for i in range(7)]]
        for i in range(7):
            ops += [
                Fs[0 if i == fpos else 1],
                [7 * cc + i for cc in range(3)] + [21 + i],
            ]
        key = fpos
        if key not in paths:
            paths[key] = oe.contract_path(*ops, list(range(21, 28)), optimize="dp")[0]
        T = oe.contract(*ops, list(range(21, 28)), optimize=paths[key])
        assert np.array_equal(T, np.rint(T))
        T = T.astype(np.int64)
        if fpos is None:
            T = (
                T.reshape(*([2, 2] * 7))
                .transpose([2 * i for i in range(7)] + [2 * i + 1 for i in range(7)])
                .reshape(128, 128)
            )
            bits = [sum(p) for p in itertools.product(range(2), repeat=7)]
            Y = np.array([T[np.array(bits) == r].sum(axis=0) for r in range(8)])
            YY[0].append(Y)
            for j in range(7):
                TX = (
                    T.reshape(*([2] * 14))
                    .transpose(
                        [i for i in range(7) if i != j] + [j] + list(range(7, 14))
                    )
                    .reshape(64, 2, 128)
                )
                b6 = [sum(p) for p in itertools.product(range(2), repeat=6)]
                Y = np.array(
                    [TX[np.array(b6) == r].sum(axis=0) for r in range(7)]
                ).reshape(14, 128)
                YY[1].append(Y)
        else:
            order = [fpos] + [i for i in range(7) if i != fpos]
            dd = [(1, 4) if i == fpos else (2, 2) for i in range(7)]
            T = (
                T.reshape(*[x for ds in dd for x in ds])
                .transpose([2 * i for i in order] + [2 * i + 1 for i in order])
                .reshape(64, 256)
            )
            b6 = [sum(p) for p in itertools.product(range(2), repeat=6)]
            YY[2].append(np.array([T[np.array(b6) == r].sum(axis=0) for r in range(7)]))
    for i in range(3):
        images[i].append(fmat(Us[i]).transpose() * fmat(Gs[i]) * fmat(np.hstack(YY[i])))
    print("CUBIC IMAGES", t, "seconds", time.time() - start, flush=True)
# exact rational coefficients of physical constraints
VN = flint.fmpq_mat([[t**j for j in range(4)] for t in range(4)]).inv()
Cimages = []
for i in range(3):
    Ci = []
    for k in range(4):
        A = images[i][0] * VN[k, 0]
        for t in range(1, 4):
            A += images[i][t] * VN[k, t]
        Ci.append(A)
    Cimages.append(Ci)
# local Fourier matrix units in rational highest-weight coordinates
units = []
for m, d in [(0, 1), (1, 2)]:
    for a in range(d):
        for b in range(d):
            E = sp.zeros(8)
            for k, O in enumerate(OPS):
                rr = sp.ones(1, 1) if m == 0 else R[PP.index(tuple(np.argsort(PP[k])))]
                E += sp.Rational(d, 6) * rr[b, a] * sp.Matrix(O)
            units.append(E)
# flip last copy and pair cosets
Eps = np.kron(np.eye(4, dtype=int), np.array([[0, 1], [-1, 0]]))


def flip(A):
    X = Eps.T @ np.array(A, object) @ Eps
    return sp.Matrix(X.reshape(*([2] * 6)).transpose(0, 1, 5, 3, 4, 2).reshape(8, 8))


def coords(A):
    return [A[0, 0]] + list(L * A * VV)


for col, unit in enumerate(units):
    assert coords(unit) == [int(col == j) for j in range(5)]
    flipped = flip(unit)
    assert sum((v * E for v, E in zip(coords(flipped), units)), sp.zeros(8)) == flipped
Phi = sp.Matrix.hstack(*[sp.Matrix(coords(flip(E))) for E in units])
assert Phi * Phi == sp.eye(5)
ctx = flint.fmpq_mpoly_ctx.get(["h0", "h1", "h2", "h3", "h4"])
hv = ctx.gens()
full = []
cos = [PP.index(p) for p in [(0, 1, 2), (0, 2, 1), (2, 1, 0)]]
for model in range(3):
    n = Us[model].shape[1]
    out = [[ctx.constant(0) for j in range(n)] for i in range(n)]
    HG = fmat(Us[model]).transpose() * fmat(Gs[model])
    UU = Us[model]
    for aa in cos:
        for bb in cos:
            # exact map h -> coordinates of Phi(P_aa h P_bb^-1)
            PL = sp.Matrix(OPS[aa])
            PR = PL if False else sp.Matrix(OPS[bb]).T
            T = sp.Matrix.hstack(*[sp.Matrix(coords(flip(PL * E * PR))) for E in units])
            z = [
                sum(
                    (flint.fmpq(str(T[i, j])) * hv[j] for j in range(5)),
                    ctx.constant(0),
                )
                for i in range(5)
            ]
            A = np.array(z[1:], object).reshape(2, 2)
            RR = (
                sy(A, 7)
                if model == 0
                else (
                    np.kron(sy(A, 6), A)
                    if model == 1
                    else np.array([[v * z[0] for v in row] for row in sy(A, 6)], object)
                )
            )
            for i in range(n):
                for j in range(n):
                    p = ctx.constant(0)
                    for a in range(RR.shape[0]):
                        for b in range(RR.shape[1]):
                            if HG[i, a] and UU[b, j]:
                                p += HG[i, a] * int(UU[b, j]) * RR[a, b]
                    out[i][j] += p / 9
    # save full polynomial rational coefficients, no kernel choice yet
    mm = mons(5, 7)
    pos = {m: i for i, m in enumerate(mm)}
    arr = [
        [[str(out[i][j].to_dict().get(m, 0)) for j in range(n)] for i in range(n)]
        for m in mm
    ]
    full.append(arr)
    print("CUBIC POLY", model, "seconds", time.time() - start, flush=True)
# exact five-copy -> three-copy partial trace maps, rational physical coordinates
left, right, orbits = cosets(2)
MI = []
MS = []
embed = np.array(s5.EMB_NUM, object)
swap = s5.OPS[s5.PINDEX[(0, 1, 2, 4, 3)]]
for a in range(10):
    for b in range(10):
        X = (
            np.array(s5.OPS[left[a]], object)
            @ embed
            @ np.array(s5.OPS[right[b]], object)
        )
        XI = np.trace(X.reshape(42, 8, 4, 8, 4), axis1=2, axis2=4)
        XS = np.trace((swap @ X).reshape(42, 8, 4, 8, 4), axis1=2, axis2=4)
        MI.append([[str(v / 120) for v in coords(sp.Matrix(Y))] for Y in XI])
        MS.append([[str(v / 120) for v in coords(sp.Matrix(Y))] for Y in XS])
obj = dict(
    U=[U.tolist() for U in Us],
    G=[[[str(v) for v in row] for row in GG] for GG in Gs],
    images=[
        [
            [[str(C[i, j]) for j in range(C.ncols())] for i in range(C.nrows())]
            for C in CC
        ]
        for CC in Cimages
    ],
    monomials=mons(5, 7),
    poly=full,
    MI=MI,
    MS=MS,
    orbits=orbits,
)
assert json.loads(json.dumps(obj)) == json.load(open(root + "/cubic.json"))
print("EXACT CUBIC SETUP DONE seconds", time.time() - start, flush=True)
