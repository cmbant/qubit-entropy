"""Exact dyadic factors of the positive five-copy baseline.

Entry (i,j) is the signed odd integer m[i,j] times 2**e[i,j].
No floating-point conversion is used to reconstruct a rational factor.
"""
import json
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"


def load_candidate():
    source = json.loads((DATA / "five_copy_twelve_face_candidate_basis.json").read_text())
    with np.load(DATA / source["basis_dyadic_file"], allow_pickle=False) as arrays:
        for face in source["faces"]:
            i = face["basis_dyadic_array"]
            K = arrays[f"k{face['kernel_integer_array']}"]
            assert K.dtype.kind == "i" and K.dtype.itemsize <= 8 and K.ndim == 2
            face["kernel_columns"] = K.tolist()
            m, e = arrays[f"m{i}"], arrays[f"e{i}"]
            assert m.dtype == np.int64 and e.dtype == np.int16
            assert m.shape == e.shape and m.ndim == 2 and m.shape[0] == m.shape[1]
            assert np.all((m == 0) | (m % 2 != 0))
            assert np.all((m > -2**53) & (m < 2**53))
            assert np.all((e >= -400) & (e <= 400))
            exponent = max(0, -int(e.min()))
            face["basis_numerator"] = [[int(m[a,b]) << (int(e[a,b])+exponent)
                                         for b in range(m.shape[1])]
                                        for a in range(m.shape[0])]
            face["basis_denominator_exponent"] = exponent
    return source["faces"]


def load_frames(path=None):
    source = json.loads(Path(path or DATA / "five_copy_repair_frames.json").read_text())
    faces = source["faces"]
    if "vectors" in faces[0]:
        return faces
    assert source["generator"] == "PCG64"
    out = []
    for face in faces:
        rng = np.random.Generator(np.random.PCG64(face["seed"]))
        vectors = rng.integers(source["low"], source["high"],
                               size=(source["vectors_per_face"], face["size"]),
                               dtype=np.int64)
        assert np.all(np.any(vectors != 0, axis=1))
        out.append(dict(tag=face["tag"], rank=face["rank"], vectors=vectors.tolist()))
    return out
