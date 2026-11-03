"""Joint trace coefficients for the sharp seven-qubit entropy certificate."""

import json, math
from pathlib import Path
import numpy as np
from flint import arb, arb_mat, ctx
from seven_entropy_coefficients import x, u, c, a, d, b, CP, f, E_star

ROOT = Path(__file__).resolve().parent.parent / "data" / "seven_entropy"


def build():
    ctx.prec = 256
    E = E_star
    features = [
        tuple(tuple(w) for w in ff)
        for ff in json.load(open(ROOT / "joint_features.json"))
    ]
    pos = {ff: i for i, ff in enumerate(features)}

    def canonical(w):
        w = tuple(w)
        return min(
            a[k:] + a[:k]
            for a in [
                w,
                w[::-1],
                tuple(1 - i for i in w),
                tuple(1 - i for i in w[::-1]),
            ]
            for k in range(len(w))
        )

    pol = [arb(0) for ff in features]
    pol[pos[()]] = E - 8 * CP[0][0]
    for i in range(6):
        for j in range(6 - i):
            if i + j == 0:
                continue
            ff = () if i + j == 1 else (canonical((0,) * i + (1,) * j),)
            pol[pos[ff]] -= CP[i][j]

    # Noncommutative trace polynomials; every absent norm factor is homogenized.
    def sadd(a, b):
        o = a.copy()
        for w, c in b.items():
            o[w] = o.get(w, arb(0)) + c
        return o

    def smul(a, b):
        o = {}
        for wa, ca in a.items():
            for wb, cb in b.items():
                w = tuple(sorted(wa + wb, key=lambda ww: (len(ww), ww)))
                o[w] = o.get(w, arb(0)) + ca * cb
        return o

    def scale(a, x):
        return {w: x * c for w, c in a.items()}

    def madd(a, b):
        o = a.copy()
        for w, c in b.items():
            o[w] = sadd(o.get(w, {}), c)
        return o

    def mmul(a, b):
        o = {}
        for wa, ca in a.items():
            for wb, cb in b.items():
                o[wa + wb] = sadd(o.get(wa + wb, {}), smul(ca, cb))
        return o

    def mscale(a, x):
        return {w: scale(c, x) for w, c in a.items()}

    def adj(a):
        return {w[::-1]: c for w, c in a.items()}

    def tr(a):
        o = {}
        for w, c in a.items():
            if not w:
                c = scale(c, 8)
            elif len(w) > 1:
                c = smul(c, {(canonical(w),): arb(1)})
            o = sadd(o, c)
        return o

    po = 6 * c * c + 2 * a * a
    pi = b * b + 5 * c * c + 2 * d * d
    alpha = (pi - po) / (c * c)
    R = {(0,): {(): arb(1)}}
    S = {(1,): {(): arb(1)}}
    EE = mscale(madd(R, S), arb(1) / 2)
    O = mscale(madd(R, mscale(S, -1)), arb(1) / 2)
    Q = [
        {(): {((0, 0),): 1 - alpha, ((0, 1),): alpha, (): -po}},
        madd(mmul(O, EE), mscale(O, -6 * u)),
        madd(mmul(EE, O), mscale(mmul(O, EE), -1)),
    ]
    choices = np.load(ROOT / "joint_choices.npz")
    Bs = []
    for cap in [False, True]:
        pp = [[[arb(0) for ff in features] for j in range(3)] for i in range(3)]
        for i in range(3):
            for j in range(3):
                A = tr(mmul(adj(Q[i]), Q[j]))
                if cap:
                    A = sadd(
                        scale(A, arb(497) / 1000),
                        scale(tr(mmul(mmul(adj(Q[i]), R), Q[j])), -1),
                    )
                for ff, cc in A.items():
                    pp[i][j][pos[ff]] += cc
        B = choices["CAP" if cap else "HS"]
        Bs.append(B)
        BA = arb_mat(B.tolist())
        QQ = BA * BA.transpose()
        for k in range(len(features)):
            for i in range(3):
                for j in range(3):
                    pol[k] -= QQ[i, j] * pp[i][j][k]
    cv = np.array([float(v.mid()) for v in pol])
    er = np.array([float(abs(v - arb(float(z))).upper()) for v, z in zip(pol, cv)]) * (
        1 + 2**-20
    )
    print("E7-1 exact joint remainder coefficient enclosures", flush=True)
    return dict(coeff=cv, radius=er)
