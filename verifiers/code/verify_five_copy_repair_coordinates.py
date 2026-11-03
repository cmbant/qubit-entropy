"""FP5-4: exact coordinate action and rounding audit for compact Gram repairs.

The text proves the norm estimate for all repairs. This checks every supplied
rational frame, reconstructs its action on independent nonsymmetric integer
probe matrices, and checks the floating-point rotation against exact values.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import flint
import numpy as np
from five_copy_repair_coordinates import exact_frames, transform
from five_copy_candidate_data import load_frames

DATA = Path(__file__).resolve().parent.parent / "data"
WIDTH = 7


def main():
    original = load_frames()
    projection = [dict(tag=f["tag"], rank=f["rank"], vectors=f["vectors"]) for f in original]
    digest = hashlib.sha256(json.dumps(projection, sort_keys=True, separators=(",", ":")).encode())
    assert digest.hexdigest() == "e3f27af96c03c30cbcfa04c99eee8c8798eec8f126938cc0f3dea6b2db84e91f"
    print("FP5-4 all 36480 original Householder integers and frame ranks preserved exactly")
    print("FP5-4 frame recipe bytes", (DATA / "five_copy_repair_frames.json").stat().st_size)
    all_frames = exact_frames(DATA / "five_copy_repair_frames.json")
    total = 0
    for index, (N, den, Q) in enumerate(all_frames):
        k = N.nrows()
        r = min(WIDTH, k)
        G = flint.fmpz_mat(k, k, [((i + 3) * (j + 5) + 7*i + j*j) % 23 - 11
                                  for i in range(k) for j in range(k)])
        U = flint.fmpz_mat(k, r, [int(N[i, j]) for i in range(k) for j in range(r)])
        M = U.transpose() * (G + G.transpose())
        raw = np.array([float(Fraction(int(v), den)) for v in M.entries()])
        # Correctly rounded rational-to-binary64 conversion, then an upward
        # Frobenius bound covering the short norm computation.
        radius = np.linalg.norm(raw) * (2.0**-53 / (1-2.0**-53)) * (1+2.0**-20)
        got, error = transform(raw, radius, [Q], WIDTH)
        T = N.transpose() * (G + G.transpose()) * N
        pairs = [(a, b) for a in range(r) for b in range(a, k)]
        expected = [Fraction(int(T[a, b]), (2 if a == b else 1)*den**2)
                    for a, b in pairs]
        assert sum((Fraction(float(v))-e)**2 for v, e in zip(got, expected)) <= Fraction(error)**2
        z = [(i*11 + 3) % 17 - 8 for i in range(len(pairs))]
        J = flint.fmpz_mat(k, k)
        for value, (a, b) in zip(z, pairs):
            J[a, b] = value
            J[b, a] = J[a, b]
        D = N * J * N.transpose()
        direct = sum(int(G[i, j]) * int(D[j, i]) for i in range(k) for j in range(k))
        assert Fraction(direct, den**2) == sum(value*e for value, e in zip(z, expected))
        assert sum(int(v)**2 for v in J.entries()) <= 2*sum(value**2 for value in z)
        # A wrong convention which omits the frame rotation must fail.
        wrong = np.array([raw.reshape(r, k)[a, b] for a, b in pairs])
        assert np.linalg.norm(wrong-got) > 1000*error
        total += len(pairs)
        print(f"FP5-4 face {index}: {k} kernel columns, {len(pairs)} coordinates; "
              "exact trace and norm, rigorous rotation radius, negative control PASS")
    assert total == 3754
    six = sum(min(6, N.nrows())*N.nrows()-min(6, N.nrows())*(min(6, N.nrows())-1)//2
              for N, den, Q in all_frames)
    assert six == 3249 < 3283
    print("FP5-4 width six has only", six, "coordinates, fewer than the critical dimension 3283")
    print("FP5-4 independent repair coordinates", total, "(previous rectangular count 8965)")
    print("FP5-4 PASS: exact frame/action audit; no global identity or rank claim")


if __name__ == "__main__":
    main()
