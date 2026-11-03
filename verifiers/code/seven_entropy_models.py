"""Exact physical kernel projection with Arb enclosures, for E7-1."""

import os, sys, json, time, pickle, math, itertools

os.environ["OPENBLAS_NUM_THREADS"] = "2"
from pathlib import Path

CODES = str(Path(__file__).resolve().parent)
import numpy as np, sympy as sp, flint
from fractions import Fraction
from five_copy_physical_faces import apply_gram, integer_matrix
from five_copy_young51_model import sym_power
from five_copy_probe_models import mons, kronvec
import verify_five_copy_site_algebra as s5
from flint import arb, arb_mat, ctx
from five_copy_candidate_data import load_frames
from seven_entropy_data import load_bases, load_constraints


def transported_basis(M, U, local):
    fs = []
    hds = []
    den = 1
    for dd, n in M["fac"]:
        T, Ci, deni, hd = local[dd]
        den *= deni**n
        if dd == 4 and n == 2:
            pairs = list(
                itertools.combinations(range(4), 2)
                if M["wedge"]
                else itertools.combinations_with_replacement(range(4), 2)
            )
            S = np.array(
                [
                    [
                        Ci[c, a] * Ci[d, b]
                        + (
                            -Ci[d, a] * Ci[c, b]
                            if M["wedge"]
                            else (Ci[d, a] * Ci[c, b] if c != d else 0)
                        )
                        for a, b in pairs
                    ]
                    for c, d in pairs
                ],
                np.int64,
            )
            hh = (
                [hd[a] * hd[b] / 2 for a, b in pairs]
                if M["wedge"]
                else [
                    hd[a] * hd[b] * (1 if a == b else Fraction(1, 2))
                    for a, b in pairs
                ]
            )
        else:
            S = Ci if n == 1 else sym_power(Ci, n)
            hh = [
                Fraction(math.prod(math.factorial(v) for v in m), math.factorial(n))
                * math.prod(hd[j] ** m[j] for j in range(dd))
                for m in mons(dd, n)
            ]
        fs.append(S)
        hds.append(hh)
    UN = apply_gram(U, fs)
    Usq = flint.fmpq_mat(integer_matrix(UN)) / den
    return Usq, hds


def build():
    ctx.prec = 256
    root = str(Path(__file__).resolve().parent.parent / "data" / "seven_entropy")
    D = load_bases()
    CC = load_constraints()
    defs = json.load(open(root + "/models.json"))
    piv = json.load(open(root + "/pivots.json"))
    start = time.time()
    from seven_entropy_coefficients import x, u

    t = (x / u).sqrt()
    ctx.prec = 256
    local = {}
    for dd, m in [(4, 1), (5, 2)]:
        L, H = (s5.V[m].T * s5.V[m]).LDLdecomposition(hermitian=False)
        T = L.T
        den = int(sp.ilcm(*[z.q for z in T]))
        local[dd] = (
            T,
            np.array(T * den, np.int64),
            den,
            [Fraction(H[i, i]) for i in range(dd)],
        )
    frames = load_frames()

    def am(A):
        if isinstance(A, flint.fmpq_mat):
            return arb_mat(A)
        return arb_mat(
            [
                [
                    (
                        arb(str(v))
                        if not isinstance(v, (float, np.floating))
                        else arb(float(v))
                    )
                    for v in row
                ]
                for row in A
            ]
        )

    def midpoint(A):
        X = np.array(
            [[float(A[i, j].mid()) for j in range(A.ncols())] for i in range(A.nrows())]
        )
        er = arb(0)
        radii = np.empty_like(X)
        for i in range(A.nrows()):
            for j in range(A.ncols()):
                ee = abs(A[i, j] - arb(float(X[i, j]))).upper()
                radii[i, j] = float(ee) * (1 + 2**-20)
                er += ee**2
        return X, float(er.sqrt().upper()) * (1 + 2**-20), radii

    def fmat(A):
        return flint.fmpq_mat([[flint.fmpq(str(v)) for v in row] for row in A])

    ms = []
    for i, M in enumerate(defs):
        assert isinstance(M["coef"], int) and M["coef"] > 0
        U = D["u%d" % i]
        Usq, hds = transported_basis(M, U, local)
        Us = arb_mat(Usq)
        hd = kronvec(hds)
        HUsq = flint.fmpq_mat(Usq)
        for a, h in enumerate(hd):
            for b in range(Usq.ncols()):
                HUsq[a, b] *= flint.fmpq(h.numerator, h.denominator)
        HUs = arb_mat(HUsq)
        S = Us.transpose() * HUs
        S_inv = S.inv()
        assert all(v.is_finite() for v in S_inv.entries())
        # Only columns selected by exact generic rank validation are required.
        J = piv[str(i)]["columns"]
        cs = CC["c%d" % i][:, :, J]
        XX = am(cs[5].tolist())
        for j in range(4, -1, -1):
            XX = t * XX + am(cs[j].tolist())
        if i in [1, 2, 3]:
            Eu = integer_matrix(U).transpose() * integer_matrix(U)
            XX = S * (arb_mat(flint.fmpq_mat(Eu).inv()) * XX)
        K = S_inv * XX
        GG = XX.transpose() * K
        Ginv = GG.inv()
        assert all(v.is_finite() for v in Ginv.entries())
        K0 = am(D["b%d" % i].tolist())
        K0 = K0 - K * (Ginv * (XX.transpose() * K0))
        W = Us * K0
        WL = HUs * K0
        k = W.ncols()
        r = min(7, k) if i < 12 else 0
        UF = flint.fmpq_mat(k, r, [int(a == b) for a in range(k) for b in range(r)])
        if r:
            for vv in frames[i]["vectors"]:
                v = flint.fmpq_mat(k, 1, [int(z) for z in vv])
                nn = sum(int(z) ** 2 for z in vv)
                UF -= 2 * v * (v.transpose() * UF) / nn
            assert UF.transpose() * UF == flint.fmpq_mat(
                r, r, [int(a == b) for a in range(r) for b in range(r)]
            )
        WU = W * arb_mat(UF)
        WLU = WL * arb_mat(UF)
        out = M.copy()
        out["fac"] = [tuple(f) for f in M["fac"]]
        out["repair_rank"] = r
        out["errors"] = {}
        out["radii"] = {}
        for name, B in [("VR", W), ("VL", WL), ("VRU", WU), ("VLU", WLU)]:
            out[name], out["errors"][name], out["radii"][name] = midpoint(B)
        assert max(out["errors"].values()) < 1e-9
        print(
            "EXACT ARB KERNEL",
            i,
            "errors",
            out["errors"],
            "sec",
            time.time() - start,
            flush=True,
        )
        ms.append(out)

    print("ALL 17 ARB KERNELS ENCLOSED seconds", time.time() - start, flush=True)

    meta = {
        "local": {
            str(dd): {"C": [[str(v) for v in row] for row in local[dd][0].tolist()]}
            for dd in local
        }
    }
    return ms, meta
