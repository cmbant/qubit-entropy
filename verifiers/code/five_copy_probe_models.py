"""Exact rational polynomial bases for the five-copy residual audit."""

import json, math, itertools, time
from pathlib import Path
import numpy as np
import sympy as sp
import flint
from fractions import Fraction
import verify_five_copy_site_algebra as s5
from five_copy_unflipped_oneflip_faces import models as othermodels

DATA = Path(__file__).resolve().parent.parent / "data"
from five_copy_physical_faces import integer_matrix, models as fourmodels, apply_gram
from five_copy_allv_young_models import invariant_basis
from five_copy_young51_model import sym_power
from five_copy_candidate_data import load_candidate, load_frames


def mons(d, n):
    return [
        tuple(c.count(i) for i in range(d))
        for c in itertools.combinations_with_replacement(range(d), n)
    ]


def kronvec(parts):
    out = [Fraction(1)]
    for part in parts:
        out = [a * b for a in out for b in part]
    return out


def build():
    frames = load_frames()
    recs = load_candidate()
    base = (
        [m[1] for m in fourmodels()]
        + [invariant_basis(6, 1), invariant_basis(5, 2)]
        + [m[4] for m in othermodels()]
    )
    local = {}
    start = time.time()
    out = []
    for d, m in [(4, 1), (5, 2)]:
        G = s5.V[m].T * s5.V[m]
        L, D = G.LDLdecomposition(hermitian=False)
        C = L.T
        den = int(sp.ilcm(*[int(v.q) for v in C]))
        Ci = np.array(C * den, dtype=np.int64)
        local[d] = dict(C=C, Ci=Ci, den=den, D=[Fraction(D[i, i]) for i in range(d)])
    for index, (rec, U) in enumerate(zip(recs, base)):
        K = flint.fmpz_mat(rec["kernel_columns"])
        B = flint.fmpz_mat([[int(v) for v in row] for row in rec["basis_numerator"]])
        raw = integer_matrix(U) * K
        k = B.nrows()
        fac = [tuple(v) for v in rec["fac"]]
        dims = [len(mons(d, n)) for d, n in fac]
        factors = []
        hd = []
        denR = 2 ** rec["basis_denominator_exponent"]
        for d, n in fac:
            ll = local[d]
            Ci = ll["Ci"]
            bound = int(np.max(abs(Ci).sum(axis=0))) ** n
            assert bound < 2**63
            if n == 1:
                SS = Ci
            else:
                assert d == 5
                SS = sym_power(Ci, n)
            factors.append(SS)
            denR *= ll["den"] ** n
            hd.append(
                [
                    Fraction(math.prod(math.factorial(v) for v in m), math.factorial(n))
                    * math.prod(ll["D"][i] ** m[i] for i in range(d))
                    for m in mons(d, n)
                ]
            )
        raw_np = np.array(
            [[int(raw[i, j]) for j in range(k)] for i in range(raw.nrows())], dtype=object
        )
        TT = apply_gram(raw_np, factors)
        nums = integer_matrix(TT) * B
        hdiag = kronvec(hd)
        VR = np.empty((nums.nrows(), k))
        VL = np.empty_like(VR)
        for i, h in enumerate(hdiag):
            for j in range(k):
                vv = int(nums[i, j])
                VR[i, j] = float(Fraction(vv, denR))
                VL[i, j] = float(Fraction(vv * h.numerator, denR * h.denominator))
        fr = frames[index]
        r = int(fr["rank"])
        UF = flint.fmpz_mat(k, r, [int(i == j) for i in range(k) for j in range(r)])
        ud = 1
        for vv in fr["vectors"]:
            v = flint.fmpz_mat(k, 1, [int(x) for x in vv])
            nn = sum(int(x) ** 2 for x in vv)
            UF = nn * UF - 2 * v * (v.transpose() * UF)
            ud *= nn
        assert UF.transpose() * UF == ud**2 * flint.fmpz_mat(
            r, r, [int(i == j) for i in range(r) for j in range(r)]
        )
        nu = nums * UF
        VRU = np.empty((nums.nrows(), r))
        VLU = np.empty_like(VRU)
        for i, h in enumerate(hdiag):
            for j in range(r):
                vv = int(nu[i, j])
                VRU[i, j] = float(Fraction(vv, denR * ud))
                VLU[i, j] = float(Fraction(vv * h.numerator, denR * ud * h.denominator))
        assert np.count_nonzero(VR) == sum(v != 0 for v in nums.entries())
        assert np.count_nonzero(VL) == np.count_nonzero(VR)
        assert np.count_nonzero(VRU) == sum(v != 0 for v in nu.entries())
        assert np.count_nonzero(VLU) == np.count_nonzero(VRU)
        nonzero = np.r_[abs(VR[VR != 0]), abs(VL[VL != 0]), abs(VRU[VRU != 0]), abs(VLU[VLU != 0])]
        assert len(nonzero) and min(nonzero) > 2**-400 and max(nonzero) < 2**400
        M = dict(
            tag=rec["tag"],
            flip=rec["flip"],
            fac=fac,
            dims=dims,
            coef=int(rec["coefficient"]),
            VR=VR,
            VL=VL,
            VRU=VRU,
            VLU=VLU,
            repair_rank=r,
        )
        out.append(M)
        print(
            "RATIONAL POLYNOMIAL MODEL",
            rec["tag"],
            VR.shape,
            "max right/left",
            abs(VR).max(),
            abs(VL).max(),
            "denbits",
            denR.bit_length(),
            "sec",
            time.time() - start,
            flush=True,
        )
    meta = dict(
        local={
            str(d): dict(C=[[str(v) for v in row] for row in local[d]["C"].tolist()]) for d in local
        }
    )
    return out, meta
