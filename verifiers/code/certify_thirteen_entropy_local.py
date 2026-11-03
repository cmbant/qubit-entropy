"""E13L: a nearby strict full-space thirteen-qubit entropy maximum.

The input is an exact Gaussian-integer graph-code vector. Small spectral
blocks give an Arb gradient enclosure. SVDs and Cholesky factors are only
proposals: independent residuals, coefficient enclosures, an analytic IEEE
roundoff budget and integer Gram residuals certify the full Hessian gap.
The paper uses only E13L-1 and E13L-2 (cor:thirteen-entropy-witness); the Hessian checks
E13L-0, E13L-3--E13L-5 and the local-maximality argument in a research note are not part of it.
No unrestricted global bound or equality classification is asserted.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import json
from functools import reduce
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations

import flint
import numpy as np
import sympy as sp
from scipy.linalg import cholesky, solve_triangular

from entropy_amplitude_hessian import EntropyAmplitudeHessian
from entropy_code_tools import basis, rank
from thirteen_fourcopy_gauss import graph13

flint.ctx.prec = 160
arb, acb = flint.arb, flint.acb
G = graph13()
data = json.loads(
    (
        Path(__file__).resolve().parent / "../data/thirteen_entropy_local.json"
    ).read_text()
)
gens = data["generators"]
assert data["graph"] == "G13" and rank(gens) == len(gens) == 5
labels = [[] for _ in range(32)]
for v in range(8192):
    syndrome = sum(((v & a).bit_count() % 2) << j for j, a in enumerate(gens))
    labels[syndrome].append(v)
labels = np.array(labels)
assert labels.shape == (32, 256)
assert len(data["real"]) == len(data["imag"]) == 256
assert all(type(x) is int for key in ["real", "imag"] for x in data[key])


def walsh(v):
    v = list(v)
    step = 1
    while step < len(v):
        for i in range(0, len(v), 2 * step):
            for j in range(i, i + step):
                a, b = v[j], v[j + step]
                v[j], v[j + step] = a + b, a - b
        step *= 2
    return v


phase = np.array(
    [
        (-1)
        ** sum(
            ((x >> i) & 1) * ((x >> j) & 1)
            for i in range(13)
            for j in range(i + 1, 13)
            if (G[i] >> j) & 1
        )
        for x in range(8192)
    ]
)
parts = []
for name in ["real", "imag"]:
    raw = [0] * 8192
    for i, a in zip(labels[0], data[name]):
        raw[int(i)] = a
    parts.append([int(s) * a for s, a in zip(phase, walsh(raw))])
normcode = sum(a * a + b * b for a, b in zip(data["real"], data["imag"]))
normphys = sum(a * a + b * b for a, b in zip(*parts))
assert normphys == 8192 * normcode
pp = [acb(a, b) / arb(normphys).sqrt() for a, b in zip(*parts)]
cc = [acb(a, b) / arb(normcode).sqrt() for a, b in zip(data["real"], data["imag"])]
assert abs(cc[0]) > arb(4) / 5 and all(abs(a) < arb(1) / 4 for a in cc[1:])
large = [int(v) for v, a in zip(labels[0], cc) if abs(a) > arb(1) / 10000]
assert rank(large) == 8
z = np.array([complex(float(a.real), float(a.imag)) for a in pp])
# Exact-state Arb Gram enclosure of all 40 phase/LU tangents.
cols = [[1j * a for a in pp]]
for site in range(13):
    signs = [1 - 2 * ((i >> site) & 1) for i in range(8192)]
    sw = [pp[i ^ (1 << site)] for i in range(8192)]
    cols.extend(
        [
            [1j * a for a in sw],
            [s * a for s, a in zip(signs, sw)],
            [1j * s * a for s, a in zip(signs, pp)],
        ]
    )
Ta = flint.acb_mat([[col[i] for col in cols] for i in range(8192)])
Gram = Ta.conjugate().transpose() * Ta
assert (
    sum(
        (
            (Gram[i, j].real - int(i == j)) * (Gram[i, j].real - int(i == j))
            for i in range(40)
            for j in range(40)
        ),
        arb(0),
    ).sqrt()
    < arb(1) / 50
)
del Ta, cols
H = EntropyAmplitudeHessian(z)
z = H.psi
assert sum(
    (
        abs(a - acb(float(b.real), float(b.imag))).upper()
        * abs(a - acb(float(b.real), float(b.imag))).upper()
        for a, b in zip(pp, z)
    ),
    arb(0),
).sqrt() < arb("1e-13")


def am(A):
    return flint.acb_mat(
        [[acb(arb(float(a.real)), arb(float(a.imag))) for a in row] for row in A]
    )


def frob(A):
    return sum((abs(x).upper() * abs(x).upper() for x in A.entries()), arb(0)).sqrt()


# Error propagation for every complex matrix product uses gamma_(8k).
# The table covers one simultaneous-column input (norm sqrt(32)<6.1).
u = F(1, 2**53)


def gamma(k):
    return k * u / (1 - k * u)


op, fn, size, coef = F(1) + F(1, 10**10), F(9), F(61, 10), F(101)
r_bt = gamma(512) * fn * size
r_c = op * r_bt + gamma(1024) * fn * (op * size + r_bt)
r_split = r_c + u * (op**2 * size + r_c)
r_d = 3 * coef * r_split + gamma(8) * 3 * coef * (op**2 * size + r_c)
dnorm = 3 * coef * op**2 * size
r_dv = op * r_d + gamma(512) * fn * (dnorm + r_d)
r_base = coef * r_bt + u * coef * (op * size + r_bt)
r_inner = r_dv + r_base + u * (op * dnorm + coef * op * size + r_dv + r_base)
r_out = op * r_inner + gamma(512) * fn * (op * dnorm + coef * op * size + r_inner)
r_sum = gamma(1716) * (op**2 * dnorm + coef * op**2 * size + r_out)
# Division, multiplier subtraction, and both unitary Walsh transforms.
# The transforms have relative error <1e-14; their input error <1e-13
# is multiplied by an operator of norm <1000.
r_column = r_out + r_sum + F(1, 10**9)
assert r_column < F(4, 10**8)
assert 23 * F(4, 10**8) < F(1, 10**6) and 512 < 23**2
assert abs(arb(float(np.sqrt(8192))) / arb(8192).sqrt() - 1) < arb("1e-15")
assert F(44 * 2100 * 3, 2) * 8 + 3 * 100 * 2 * 6 + 48 * 36 < 2 * 10**6
assert F(1, 10**5) - F(2 * 10**6, 10**12) > F(1, 2 * 10**5)
assert F(1, 10**18) < F(1, 2 * 10**5) * F(1, 10**12) / 2
assert F(1, 80000) - F(1, 400000) == F(1, 100000)
# Independent exact density-matrix Hessians at a rectangular 2x3 input.
ell = sp.Symbol("ell", positive=True)
M0 = sp.Matrix([[sp.Rational(1, 2), 0, 0], [0, sp.Rational(1, 4), 0]])
logs0 = sp.diag(-2 * ell, -4 * ell)
Loewner = sp.Matrix([[4, 32 * ell / 3], [32 * ell / 3, 16]])
e12 = sp.Matrix([[0, 1, 0], [0, 0, 0]])
e21 = sp.Matrix([[0, 0, 0], [1, 0, 0]])
controls = [
    (e12, sp.Rational(8, 3) - 2 / ell),
    (sp.I * e12, sp.Rational(8, 3) - 2 / ell),
    (e12 + e21, -4 / ell),
    (e12 - e21, sp.Rational(32, 3) - 4 / ell),
    (sp.Matrix([[0, 0, 1], [0, 0, 0]]), 4 - 2 / ell),
]
for direction, expected in controls:
    dr = direction * M0.conjugate().T + M0 * direction.conjugate().T
    direct = (
        -(
            2 * sp.trace(direction.conjugate().T * (logs0 + sp.eye(2)) * direction)
            + sp.trace(dr * Loewner.multiply_elementwise(dr))
        )
        / ell
    )
    assert sp.simplify(direct - expected) == 0
xx = sp.Symbol("xx", positive=True)
assert sp.simplify(sp.diff(-(xx**2) * sp.log(xx**2) / ell, xx, 3) + 4 / (xx * ell)) == 0
print(
    "E13L-0 PASS: exact roundoff, amplitude-derivative and local-existence constants.",
    flush=True,
)

words = []
for x in gens:
    zz = 0
    for j in range(13):
        if x >> j & 1:
            zz ^= G[j]
    words.append((x, zz))
assert (
    min(
        (x | zz).bit_count()
        for mask in range(1, 32)
        for x, zz in [
            (
                reduce(int.__xor__, [words[j][0] for j in range(5) if mask >> j & 1]),
                reduce(int.__xor__, [words[j][1] for j in range(5) if mask >> j & 1]),
            )
        ]
    )
    == 8
)
ident = flint.acb_mat(np.eye(64, dtype=int).tolist())
allidx = np.arange(8192).reshape([2] * 13)
ln2 = arb(2).log()
hh, etot, error, counts = [acb(0) for _ in pp], arb(0), arb(0), {1: 0, 2: 0}
for count, (cut, record) in enumerate(zip(combinations(range(13), 6), H.cache), 1):
    q, iq, U, s, Vh, k, aa, bb = record
    # Row bit j in this flattening is cut[5-j].
    restrictions = [
        sum(((x >> site) & 1) << j for j, site in enumerate(cut[::-1]))
        | (sum(((zz >> site) & 1) << j for j, site in enumerate(cut[::-1])) << 6)
        for x, zz in words
    ]
    V, d, e, c, blocks = basis(restrictions, 6)
    assert (e, c) in [(1, 3), (2, 1)]
    counts[e] += 1
    ix = allidx.transpose(q).reshape(64, 128)
    M = flint.acb_mat([[pp[int(j)] for j in row] for row in ix])
    Vm = am(V)
    Cp = Vm.conjugate().transpose() * M
    Lall, ec = flint.acb_mat(64, 64), acb(0)
    for b in blocks:
        B = flint.acb_mat([[Cp[i, j] for j in range(128)] for i in b])
        D = B * B.conjugate().transpose() / d
        ev, Right = D.eig(right=True)
        diag = flint.acb_mat(len(b), len(b))
        for j, t in enumerate(ev):
            assert t.real > arb(1) / 2100**2
            diag[j, j] = t.log()
            ec -= t * t.log()
        L = Right * diag * Right.inv()
        for bell in range(1 << e):
            for j, a in enumerate(b):
                for j1, a1 in enumerate(b):
                    Lall[a + bell, a1 + bell] = L[j, j1]
    hc = -(Vm * Lall * Cp) / (d * ln2)
    for i, row in enumerate(ix):
        for j, idx in enumerate(row):
            hh[int(idx)] += hc[i, j]
    etot += (1 << e) * ec.real / ln2

    # Independent SVD residuals and exact coefficient enclosures.
    Um, Wm = am(U), am(Vh.conj().T)
    sm = [arb(float(t)) for t in s]
    sig = flint.acb_mat(64, 64)
    for i, t in enumerate(sm):
        sig[i, i] = t
    gu = frob(Um.conjugate().transpose() * Um - ident)
    gv = frob(Wm.conjugate().transpose() * Wm - ident)
    res = frob(M - Um * sig * Wm.conjugate().transpose())
    assert gu < arb("1e-10") and gv < arb("1e-10")
    du, dv = gu / (1 + (1 - gu).sqrt()), gv / (1 + (1 - gv).sqrt())
    normu, normv = (1 + gu).sqrt(), (1 + gv).sqrt()
    delta = res + du * sm[0] * normv + dv * sm[0]
    assert sm[-1] - delta > arb(1) / 2100
    e1 = 44 * delta / ((sm[-1] - delta) * ln2)
    logs = [t.log() for t in sm]
    for i in range(64):
        ki = -2 * (2 * logs[i] + 1) / ln2
        assert abs(ki - arb(float(k[i]))) < arb("1e-10") and abs(ki) < 100
        for j in range(i, 64):
            t = (sm[i] - sm[j]) / (sm[i] + sm[j])
            ratio = arb(1) if s[i] == s[j] else t.atanh() / t
            aval = -2 * (logs[i] + logs[j] + 1 + 2 * ratio) / ln2
            bval = -2 * (logs[i] + logs[j] + 1 + 2 * t * t.atanh()) / ln2
            assert abs(aval) < 100 and abs(bval) < 100
            for a1, b1 in [(aa[i, j], bb[i, j]), (aa[j, i], bb[j, i])]:
                assert abs(aval - arb(float(a1))) < arb("1e-10")
                assert abs(bval - arb(float(b1))) < arb("1e-10")
    dc = du * normv + dv
    e2 = 100 * du * (normu + 1) + 200 * dc * (normu * normv + 1)
    e3 = arb("1e-10") * normu**2 + arb("3e-10") * (normu * normv) ** 2
    error += e1 + e2 + e3
    if count % 256 == 0:
        print(
            f"E13L-1: {count} exact Pauli decompositions, Arb spectra/gradient and SVD residuals.",
            flush=True,
        )
assert count == 1716 and counts == {1: 312, 2: 1404}
E = etot / 1716
hg = walsh([int(s) * a for s, a in zip(phase, hh)])
grad = [
    2 * (hg[int(i)] / (1716 * arb(8192).sqrt()) - E * a) for i, a in zip(labels[0], cc)
]
gnorm = sum((abs(a).upper() * abs(a).upper() for a in grad), arb(0)).sqrt()
assert gnorm < arb("1e-18")
assert E > arb("5.83360583836270199")
assert E + arb("1e-21") < arb("5.83360583836270200")
z12 = sum(
    (
        (1 if i < 4096 else -1) * (a.real * a.real + a.imag * a.imag)
        for i, a in enumerate(pp)
    ),
    arb(0),
)
assert arb(7) / 1000 < z12 < arb(8) / 1000
lagerr = 2 * (abs(arb(float(H.E)) - E) + abs(1 / ln2 - arb(float(1 / np.log(2.0)))))
representation = 128 * (error / 1716 + lagerr)
assert representation < arb("1e-6")
print(
    f"E13L-1 PASS: cut types {counts}; all reduced eigenvalues >1/2100^2.", flush=True
)
print(f"E13L-2 PASS: E6={E}; full real gradient norm={gnorm}; <Z12>={z12}.", flush=True)
print(
    f"E13L-3 PASS: assembled representation error <{representation}; arithmetic budget 1e-6.",
    flush=True,
)


def fwht(v):
    v = v.copy()
    for j in range(13):
        t = v.reshape(-1, 2, 1 << j)
        a, b = t[:, 0].copy(), t[:, 1].copy()
        t[:, 0], t[:, 1] = a + b, a - b
    return v / np.sqrt(8192)


def toblocks(v):
    c = fwht(phase * v)[labels]
    return np.c_[c.real, c.imag]


def fromblocks(v):
    c = np.zeros(8192, complex)
    c[labels] = v[:, :256] + 1j * v[:, 256:]
    return phase * fwht(c)


B = np.empty((32, 512, 512))
for j in range(512):
    v = np.zeros((32, 512))
    v[:, j] = 1
    B[:, :, j] = toblocks(H.action(fromblocks(v)))
    if j % 128 == 127:
        print(f"E13L-4: assembled {j+1}/512 simultaneous Hessian columns.", flush=True)
B = (B + B.swapaxes(1, 2)) / 2
idx = np.arange(8192)
vectors = [1j * z]
vector_sectors = [0]
for i in range(13):
    sign, sw = 1 - 2 * ((idx >> i) & 1), z[idx ^ (1 << i)]
    vectors.extend([1j * sw, sign * sw, 1j * sign * z])
    for px, pz in [(1, 0), (1, 1), (0, 1)]:
        syndrome = 0
        for j, (x, zz) in enumerate(words):
            anticommute = (px * ((zz >> i) & 1)) ^ (pz * ((x >> i) & 1))
            syndrome |= anticommute << j
        vector_sectors.append(syndrome)
penalty = np.zeros_like(B)
for t, s in zip(vectors, vector_sectors):
    w = toblocks(t)
    assert np.linalg.norm(np.delete(w, s, axis=0)) < 1e-12
    penalty[s] += np.outer(w[s], w[s])
A = -B + penalty
# Penalty/Gram and 2^-44 grid error together have norm <1e-8.
assert representation + arb("1e-6") + arb("1e-8") < arb(1) / 400000


def integer_norm2(A):
    return sum(int(t) ** 2 for t in A.entries())


def residual_test(K, Li, Ji, scale):
    n = K.nrows()
    identity = flint.fmpz_mat(n, n)
    for i in range(n):
        identity[i, i] = scale * scale
    R, S = scale * K - Li * Li.transpose(), identity - Ji * Li
    r = arb(integer_norm2(R)).sqrt() / scale**2
    t = arb(integer_norm2(S)).sqrt() / scale**2
    j2 = arb(integer_norm2(Ji)) / scale**2
    return t < 1 and j2 * r / (1 - t) ** 2 < 1


# Exact positive and indefinite controls, with arbitrary proposed factors.
I2 = flint.fmpz_mat([[1, 0], [0, 1]])
assert residual_test(I2, I2, I2, 1)
assert not residual_test(flint.fmpz_mat([[1, 0], [0, -1]]), I2, I2, 1)
scale = 1 << 44


def imat(a):
    assert np.max(abs(a * scale)) < 2**62
    return flint.fmpz_mat(np.rint(a * scale).astype(np.int64).tolist())


for s, aa in enumerate(A):
    K = imat(aa)
    assert K == K.transpose()
    shift = (scale + 79999) // 80000
    for i in range(512):
        K[i, i] -= shift
    numeric = np.array(K.tolist(), float) / scale
    L = cholesky(numeric, lower=True, check_finite=False)
    J = solve_triangular(L, np.eye(512), lower=True, check_finite=False)
    assert residual_test(K, imat(L), imat(J), scale)
print(
    "E13L-4 PASS: all 32 rounded penalty matrices >1/80000 by integer Gram residuals.",
    flush=True,
)
print(
    "E13L-5 PASS: exact-center penalized Hessian >1/100000; gradient <1e-18.",
    flush=True,
)
print(
    "E13L CONCLUSION: a full-rank strict local entropy maximum modulo LU lies within 2e-12 of the exact input; 5.83360583836270199<E6<5.83360583836270200.",
    flush=True,
)
print(
    "E13L SCOPE: global optimality, global equality orbits and the thirteen-qubit purity inequality remain open.",
    flush=True,
)
