"""E7-1 compact-input and dense-evaluator regression (not a global proof).

Exact fingerprints preserve the physical bases and every family coefficient.
The old dense evaluator independently recomputes all 17 face contributions
and rectangular repair coefficients on probes of both kinds. No floating
comparison here substitutes for E7-1's complete validated inverse.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import tempfile
import time
from pathlib import Path
import numpy as np
import seven_entropy_compact_evaluation as compact
import seven_entropy_evaluation as dense
from seven_entropy_models import build
from seven_entropy_data import ROOT, load_bases, load_constraints
from five_copy_repair_coordinates import frames, transform
import certify_seven_entropy_repair as certificate


def fingerprint(data):
    digest = hashlib.sha256()
    for key in sorted(data):
        a = data[key].astype("<i8")
        digest.update(key.encode() + b"\0" + str(a.shape).encode() + b"\0" + a.tobytes())
    return digest.hexdigest()


def main():
    start = time.monotonic()
    bases, constraints = load_bases(), load_constraints()
    assert fingerprint({key: a for key, a in bases.items() if key.startswith("u")}) == (
        "9a78751b617d7193be84c9d30f22794a67fe5138a0c7ea9b66b82a37fc59c16d")
    assert fingerprint(constraints) == "f2a79fbed6b6fb1c17c71d996d19b6101f02852abbf6ac837b0e71dc9a9e6745"
    for i in range(17):
        b = bases[f"b{i}"]
        assert b.dtype == np.float64 and np.isfinite(b).all()
        assert b.shape[0] == bases[f"u{i}"].shape[1]
        assert np.all(np.diag(b) > 0) and not np.any(np.triu(b, 1))
    print("E7 compact inputs PASS: all original integer bases and full family-image arrays exact")
    print("E7 compact inputs PASS: 17 lower-trapezoidal dyadic target factors")
    print("E7 compact input bytes", sum(path.stat().st_size for path in ROOT.iterdir()))
    models, meta = build()
    compact.prepare(models)
    maps = compact.local_maps(meta)
    Qs = frames(ROOT.parent / "five_copy_repair_frames.json")
    for M in models:
        for key in ("VR", "VL", "VRU", "VLU"):
            M["A" + key] = (abs(M[key]) + M["radii"][key] / (64 * compact.UROUND)) * (1 + 2.0**-40)
    with np.load(ROOT.parent / "five_copy_repair_probes.npz", allow_pickle=False) as pp:
        # Non-scalar probes from the symmetry-reduced and full-coset sets.
        for i in (0, 1700, 3400, 3423):
            num, den, kind = pp["inputs"][i].astype(np.int64), int(pp["denominator"]), int(pp["kinds"][i])
            z, er, tr, te = compact.symbol(models, maps, num, den, kind)
            y, yr, ref, re = dense.symbol(models, maps, num, den, kind)
            assert np.linalg.norm(z - y) <= er + np.linalg.norm(yr)
            assert abs(tr - ref) <= te + re
            packed, radius = transform(z, float(er), Qs, 7)
            assert packed.shape == (3754,) and np.isfinite(radius) and radius >= 0
            print(f"E7 compact/dense probe {i} kind {kind} PASS: all 17 faces; 3754 coordinates")
            # Sensitivity control for the two new F^2 sources, including the
            # exterior sign: a deliberately wrong baseline trace is rejected.
            if i == 3400:
                for j in (12, 13):
                    _, _, value, bound = compact.symbol([models[j]], maps, num, den, kind)
                    _, _, exact_ref, ref_bound = dense.symbol([models[j]], maps, num, den, kind)
                    assert abs(value - exact_ref) <= bound + ref_bound
                    assert abs(-value - exact_ref) > bound + ref_bound
                    print(f"E7 face {j} trace-sign negative control REFUTED")
    with tempfile.TemporaryDirectory(prefix="e71-checkpoint-") as directory:
        path = Path(directory) / "row.npz"
        certificate.FINGERPRINT = "test fingerprint"
        good = dict(row=np.zeros(3754), row_radius=0., residual=0., residual_radius=0.,
                    repair_width=7, fingerprint=certificate.FINGERPRINT)
        np.savez(path, **good)
        assert certificate.complete_checkpoint(path)
        for key, bad in (("fingerprint", "stale"), ("repair_width", 16),
                         ("row", np.zeros(8965)), ("row_radius", -1.),
                         ("residual_radius", np.nan)):
            np.savez(path, **{**good, key: bad})
            assert not certificate.complete_checkpoint(path)
        path.write_bytes(b"truncated checkpoint")
        assert not certificate.complete_checkpoint(path)
    print("E7 checkpoint PASS: stale sources, widths, shapes, radii and corrupt archives rejected")
    print("E7 compact regression PASS; seconds", time.monotonic() - start)


if __name__ == "__main__":
    main()
