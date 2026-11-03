"""Exact three-copy kernel localizers for the seven-qubit entropy certificate."""

import itertools, math, json, time
from pathlib import Path
import numpy as np, scipy.linalg as la, flint
from flint import arb, arb_mat, ctx
from seven_entropy_coefficients import x, u

ROOT = Path(__file__).resolve().parent.parent / "data" / "seven_entropy"


def build():
    D = json.loads((ROOT / "cubic.json").read_text())
    columns = json.loads((ROOT / "cubic_columns.json").read_text())
    mon = np.array(D["monomials"])
    t = (x / u).sqrt()
    ctx.prec = 256
    Ks = []
    Kfloat = []
    kernelpolys = []
    fullpoly = []
    start = time.time()
    ranks = [2, 2, 3]

    def qm(A):
        return flint.fmpq_mat([[flint.fmpq(str(v)) for v in row] for row in A])

    def am(A):
        return arb_mat(qm(A))

    def mid(A):
        return np.array(
            [[float(A[i, j].mid()) for j in range(A.ncols())] for i in range(A.nrows())]
        )

    Ks = []
    Kfloat = []
    kernelpolys = []
    fullpoly = []
    start = time.time()
    ranks = [2, 2, 3]
    for i, r in enumerate(ranks):
        U = am(D["U"][i])
        GG = am(D["G"][i])
        S = U.transpose() * GG * U
        Ci = [qm(c) for c in D["images"][i]]
        n = Ci[0].nrows()
        nc = Ci[0].ncols()
        for tt in range(3 * (r + 1) + 1):
            X = Ci[3]
            for k in range(2, -1, -1):
                X = tt * X + Ci[k]
            assert X.rank() <= r
        XX = arb_mat(Ci[3])
        for k in range(2, -1, -1):
            XX = t * XX + arb_mat(Ci[k])
        _, _, piv = la.qr(mid(XX), pivoting=True, mode="economic")
        JJ = list(piv[:r])
        X = arb_mat([[XX[a, int(b)] for b in JJ] for a in range(n)])
        K = S.inv() * X
        P = (
            arb_mat(np.eye(n, dtype=int).tolist())
            - K * (X.transpose() * K).inv() * X.transpose()
        )
        JJ = columns[i]
        K = arb_mat([[P[a, int(b)] for b in JJ] for a in range(n)])
        Ks.append(K)
        Kfloat.append(mid(K))
        poly = np.array(D["poly"][i])
        cf = []
        for m in range(len(mon)):
            B = K.transpose() * am(poly[m].tolist()) * K
            cf.append(B)
        vals = np.array(
            [
                [
                    float(((B[a, b] + B[b, a]) / 2).mid())
                    for a, b in zip(*np.triu_indices(n - r))
                ]
                for B in cf
            ]
        )
        kernelpolys.append(vals)
        fullpoly.append(cf)
        print("EXACT CUBIC KERNEL", i, "rank", r, "columns", JJ, flush=True)

    BASE = np.load(ROOT / "cubic_baseline.npz")
    outcoef = []
    outbound = []
    for sign in range(2):
        vals = []
        for m in range(len(mon)):
            val = arb(0)
            for j in range(3):
                B = arb_mat(BASE["b%d" % (3 * sign + j)].tolist())
                X = B.transpose() * fullpoly[j][m] * B
                for a in range(X.nrows()):
                    val += X[a, a]
            vals.append(val)
        midco = np.array([float(v.mid()) for v in vals])
        er = np.array(
            [float(abs(v - arb(float(z))).upper()) for v, z in zip(vals, midco)]
        ) * (1 + 2**-20)
        outcoef.append(midco)
        outbound.append(er)
    P = {}
    for m, up0, lo0, ue, le in zip(
        mon, outcoef[0], outcoef[1], outbound[0], outbound[1]
    ):
        up = arb(float(up0), float(ue))
        lo = arb(float(lo0), float(le))
        m = tuple(int(v) for v in m)
        for ex, cc in [
            (m + (0,) * 5, (arb(8693) / 64000 * up - arb(121) / 917 * lo) / 2),
            ((0,) * 5 + m, (arb(8693) / 64000 * up - arb(121) / 917 * lo) / 2),
        ]:
            P[ex] = P.get(ex, arb(0)) + cc
        for k in itertools.product(*[range(v + 1) for v in m]):
            if sum(k) not in [3, 4]:
                continue
            ex = tuple(a - b for a, b in zip(m, k)) + k
            cc = (lo - up) * math.prod(math.comb(a, b) for a, b in zip(m, k)) / 70
            P[ex] = P.get(ex, arb(0)) + cc
    mm = np.array(list(P), np.int64)
    cc = np.array([float(v.mid()) for v in P.values()])
    rr = np.array(
        [float(abs(v - arb(float(c))).upper()) for v, c in zip(P.values(), cc)]
    ) * (1 + 2**-20)
    vals = [
        flint.fmpq(v)
        for name in ["MI", "MS"]
        for X in D[name]
        for row in X
        for v in row
    ]
    den = math.lcm(*[int(v.denom()) for v in vals])
    maps = {}
    for name in ["MI", "MS"]:
        maps[name] = np.array(
            [[[int(flint.fmpq(v) * den) for v in row] for row in X] for X in D[name]],
            np.int64,
        ).transpose(0, 2, 1)
    print(
        "E7-K3 exact cubic kernels and localized polynomial enclosed",
        len(mm),
        "terms",
        flush=True,
    )
    return dict(
        monomials=mm,
        coeff=cc,
        radius=rr,
        denominator=den,
        orbits=np.array(D["orbits"], np.int64),
        **maps
    )
