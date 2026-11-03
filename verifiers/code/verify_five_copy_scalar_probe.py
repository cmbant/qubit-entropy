"""FP5-2: independent exact scalar-probe audit of the rounded evaluator.

For G supported on the local spin-5/2 sector, only the two-flip [7] and
one-F [6] faces survive. Their physical matrices are exact rank-one forms.
This checks all 8965 thin coefficients and the baseline trace, including
zero faces, against rational values. It is not the full dual certificate.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json
from fractions import Fraction
import numpy as np
import flint
from five_copy_probe_models import build, DATA
from five_copy_probe_evaluation import local_maps, symbol
import five_copy_compact_evaluation as compact
from five_copy_physical_faces import models as fourmodels
from five_copy_candidate_data import load_candidate, load_frames


def main():
    probes = np.load(DATA / "five_copy_repair_probes.npz")
    assert probes["inputs"].shape == (3424, 2, 42)
    assert int(probes["denominator"]) == 8192
    assert abs(probes["inputs"]).max() < 2**20
    assert np.all((probes["exponents"] >= 0) & (probes["exponents"] <= 32))
    print("FP5-2 probe denominators and dyadic scaling range checked exactly")
    candidate = load_candidate()
    frames = load_frames()
    raw_bases = fourmodels()
    expected = []
    trace = Fraction(0)
    for index, rec in enumerate(candidate):
        k = len(rec["basis_numerator"])
        rank = frames[index]["rank"]
        if index not in (0, 1):
            expected.extend([Fraction(0)] * (rank * k))
            continue
        assert (rec["flip"], rec["tag"]) == (2, "7" if index == 0 else "6")
        U = raw_bases[index][1]
        # The right functionals are e5^7 and (e3+e4) tensor e5^6.
        w = U[-1] if index == 0 else U[2 * 210 + 209] + U[3 * 210 + 209]
        K = flint.fmpz_mat(rec["kernel_columns"])
        B = flint.fmpz_mat([[int(x) for x in row] for row in rec["basis_numerator"]])
        num = flint.fmpz_mat(1, len(w), [int(x) for x in w]) * K * B
        alpha = 128 * 3**7 if index == 0 else 7 * 4 * 2**6 * 3**6 // 2
        den = 2 ** (2 * rec["basis_denominator_exponent"])
        trace += Fraction(alpha * sum(int(x) ** 2 for x in num.entries()), den)
        Q = flint.fmpz_mat(k, rank, [int(i == j) for i in range(k) for j in range(rank)])
        qden = 1
        for v in frames[index]["vectors"]:
            col = flint.fmpz_mat(k, 1, v)
            norm = sum(x * x for x in v)
            Q = norm * Q - 2 * col * (col.transpose() * Q)
            qden *= norm
        q = num * Q
        expected.extend(
            Fraction(2 * alpha * int(q[0, i]) * int(num[0, j]), den * qden)
            for i in range(rank)
            for j in range(k)
        )
    models, meta = build()
    inputs = np.zeros((2, 42), np.int64)
    inputs[0, 0] = 1
    row, radius, value, error = symbol(models, local_maps(meta), inputs, 1, 4)
    assert len(row) == len(expected) == 8965
    for center, bound, exact in zip(row, radius, expected):
        assert abs(Fraction(float(center)) - exact) <= Fraction(float(bound))
    assert abs(Fraction(float(value)) - trace) <= Fraction(float(error))
    row2, radius2, value2, error2 = symbol(models, local_maps(meta), inputs, 1, 4,
                                         chunks=(64, 256, 64))
    for center, bound, exact in zip(row2, radius2, expected):
        assert abs(Fraction(float(center))-exact) <= Fraction(float(bound))
    assert abs(Fraction(float(value2))-trace) <= Fraction(float(error2))
    print("FP5-2 wider (64,256,64) contractions: all exact scalar coefficients and trace PASS")
    compact.prepare(models)
    row3, radius3, value3, error3 = compact.symbol(models, local_maps(meta), inputs, 1, 4)
    assert sum((Fraction(float(center))-exact)**2 for center, exact in zip(row3, expected)) <= Fraction(float(radius3))**2
    assert abs(Fraction(float(value3))-trace) <= Fraction(float(error3))
    print("FP5-2 compact norm companions: exact Frobenius coefficient error and trace PASS")
    target = Fraction(796, 917) * 6**7
    print("FP5-2 exact scalar Gram factors", 128 * 3**7, 7 * 4 * 2**6 * 3**6 // 2)
    print("FP5-2 all 8965 coefficient errors and baseline trace inside proved radii")
    print("FP5-2 normalized baseline residual", float((target - trace) / 6**7))
    print("FP5-2 PASS: independent rational scalar-probe check; no global rank claim")


if __name__ == "__main__":
    main()
