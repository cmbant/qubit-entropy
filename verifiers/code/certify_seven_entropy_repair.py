"""E7-1: verified real positive repair proving the sharp seven-qubit entropy bound.

No retained fitted moment maps or SDP solver are used. Exact kernels and source
coefficients are enclosed with Arb. A floating SVD only proposes dyadic maps;
forward error bounds validate the inverse and the positive Gram correction.
E7-K0--E7-K3 establish the physical source data used here; DP5-1 supplies
the invariant-space dimension. The proof combines these checks.
"""

import os

for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import argparse, math, multiprocessing as mp, time, tempfile
import hashlib, sys, zipfile, zlib
from pathlib import Path
from contextlib import ExitStack
import numpy as np, scipy.linalg as la
import seven_entropy_compact_evaluation as p
from five_copy_repair_coordinates import frames, transform
import seven_entropy_source_evaluation as s
from seven_entropy_models import build as physical_models
from seven_entropy_cubic import build as cubic_localizer
from seven_entropy_joint import build as joint_coefficients
from seven_entropy_joint_maps import build as joint_maps
from seven_entropy_coefficients import x, E_star

DATA = Path(__file__).resolve().parent.parent / "data"
RANK = 3283
REPAIR_WIDTH = 7
FINGERPRINT = None
UROUND = 2.0**-53
INFLATE = 1 + 2.0**-20


def check_probe_inputs():
    global FINGERPRINT
    with np.load(DATA / "five_copy_repair_probes.npz", allow_pickle=False) as source:
        digest = hashlib.sha256()
        for key, dtype in [("inputs", "<i4"), ("kinds", "u1"),
                           ("denominator", "<i8"), ("exponents", "<i2")]:
            digest.update(source[key].astype(dtype).tobytes())
        assert digest.hexdigest() == "0d412648b7cf2177fa52d738f88c45778973679c35f2eb6c69ec604d5921748f"
        inputs, kinds, exponents = (source[key] for key in ("inputs", "kinds", "exponents"))
        assert inputs.shape == (3424, 2, 42) and int(source["denominator"]) == 8192
        assert not np.any(inputs[:, 1]) and np.max(abs(inputs)) < 2**20
        assert np.all((kinds == 4) | (kinds == 5))
        assert np.all((exponents >= 0) & (exponents <= 32))
        for values, kind in zip(inputs[:, 0], kinds):
            if kind == 4:
                for m, d, offset in [(0, 1, 0), (1, 4, 1), (2, 5, 17)]:
                    h = values[offset:offset+d*d].astype(np.int64).reshape(d, d)
                    R = np.array(p.s5.RHO[m][p.s5.PINDEX[(1, 0, 2, 3, 4)]], np.int64)
                    assert np.array_equal(h @ R, R @ h)
    print("E7-1 original probe arrays, denominators, ranges and coset commutations checked exactly", flush=True)
    names = ("five_copy_repair_frames.json", "five_copy_repair_probes.npz")
    digest = hashlib.sha256()
    # Bind the actual imported local helpers, rather than unrelated verifiers.
    # Keep the hash in checkpoints, not the log: public bundles may rewrite
    # docstrings or ship additional scripts without changing the arithmetic.
    directory = Path(__file__).resolve().parent
    sources = {Path(module.__file__).resolve() for module in tuple(sys.modules.values())
               if getattr(module, "__file__", None)
               and Path(module.__file__).suffix == ".py"
               and Path(module.__file__).resolve().parent == directory}
    for path in sorted(sources) + [DATA/name for name in names] + sorted((DATA / "seven_entropy").iterdir()):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    FINGERPRINT = digest.hexdigest()
    print("E7-1 saved input bytes", sum(path.stat().st_size for path in (DATA / "seven_entropy").iterdir()), flush=True)


def gamma(n):
    return n * UROUND / (1 - n * UROUND)


def upper_norm(x):
    assert x.size < 2**29 and np.isfinite(x).all()
    return np.linalg.norm(x) * INFLATE


def complete_checkpoint(path):
    try:
        with np.load(path, allow_pickle=False) as row:
            return (str(row["fingerprint"]) == FINGERPRINT
                    and int(row["repair_width"]) == REPAIR_WIDTH
                    and row["row"].shape == (3754,)
                    and all(row[key].shape == () for key in
                            ("row_radius", "residual", "residual_radius"))
                    and all(np.isfinite(row[key]).all() for key in
                            ("row", "row_radius", "residual", "residual_radius"))
                    and float(row["row_radius"]) >= 0 and float(row["residual_radius"]) >= 0)
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, zlib.error):
        return False


def initialize(directory, resume):
    global models, maps, probes, out, reuse, repair_frames
    out, reuse = Path(directory), resume
    models, meta = physical_models()
    p.prepare(models)
    repair_frames = frames(DATA / "five_copy_repair_frames.json")
    maps = p.local_maps(meta)
    with np.load(DATA / "five_copy_repair_probes.npz", allow_pickle=False) as source:
        probes = {key: source[key] for key in source.files}
    lc, jr, jm = cubic_localizer(), joint_coefficients(), joint_maps()
    s.initialize(jm, jr, lc, probes)


def evaluate(i):
    dest = out / (str(i) + ".npz")
    if reuse and complete_checkpoint(dest):
        return i
    num = probes["inputs"][i].astype(np.int64)
    den = int(probes["denominator"])
    row, rad, tr, te = p.symbol(models, maps, num, den, int(probes["kinds"][i]))
    scale = 2.0 ** int(probes["exponents"][i])
    row, er = transform(row / scale, float(rad) / scale, repair_frames, REPAIR_WIDTH)
    tr, te = tr / scale, te / scale
    sv, se, *_ = s.run(i)
    r = sv - tr
    rr = (se + te + UROUND * (abs(sv) + abs(tr) + abs(r))) * INFLATE
    temporary = dest.with_suffix(".tmp.npz")
    np.savez(temporary, row=row, row_radius=er, residual=r, residual_radius=rr,
             fingerprint=FINGERPRINT, repair_width=REPAIR_WIDTH)
    temporary.replace(dest)
    return i


def verify_repair(directory, count):
    rows = []
    for i in range(count):
        path = Path(directory) / f"{i}.npz"
        assert complete_checkpoint(path), f"incompatible checkpoint: {path}"
        with np.load(path, allow_pickle=False) as source:
            rows.append({key: source[key] for key in source.files})
    A = np.array([d["row"] for d in rows])
    r = np.array([d["residual"] for d in rows])
    er = np.array([d["row_radius"] for d in rows])
    rr = np.array([d["residual_radius"] for d in rows])
    rowexp = np.rint(np.log2(np.maximum(la.norm(A, axis=1), 2.0**-300))).astype(int)
    sc = 2.0**rowexp
    A /= sc[:, None]
    r /= sc
    er /= sc
    rr /= sc
    assert A.shape == (3424, 3754)
    assert np.all(er >= 0) and np.all(rr >= 0)
    for values in [A, r, er, rr]:
        p.check_range(values)
    start = time.time()
    U, singular, V = la.svd(A, full_matrices=False)
    U = np.ascontiguousarray(U[:, :RANK])
    V = np.ascontiguousarray(V[:RANK])
    d = 1 / singular[:RANK]
    for B in [U, V, d]:
        p.check_range(B)
    vbound = 1 + 2.0**-12
    vn, un, an = upper_norm(V), upper_norm(U), upper_norm(A)
    gram_error = (
        upper_norm(V @ V.T - np.eye(RANK)) + gamma(A.shape[1]) * vn * vn
    ) * INFLATE
    assert gram_error < vbound * vbound - 1
    H = d[:, None] * (U.T @ A)
    M = H @ V.T
    arithmetic = (
        (gamma(count) + UROUND + gamma(count) * UROUND) * max(d) * un * an * vbound
        + gamma(A.shape[1]) * upper_norm(H) * vn
    ) * INFLATE
    input_error = upper_norm(d * (abs(U).T @ er)) * vbound * INFLATE
    delta = (upper_norm(np.eye(RANK) - M) + arithmetic + input_error) * INFLATE
    projected = d * (U.T @ r)
    residual_error = d * (abs(U).T @ (rr + gamma(count) * abs(r)))
    residual_error += UROUND * abs(projected) / (1 - UROUND)
    residual_bound = upper_norm(abs(projected) + residual_error * INFLATE) * INFLATE
    assert delta < 1
    correction = 1.5 * vbound * residual_bound / (1 - delta) * INFLATE
    print("E7-1 matrix", A.shape, "validated rank", RANK, flush=True)
    print(
        "E7-1 inverse defect upper", delta, "evaluation part", input_error, flush=True
    )
    print("E7-1 projected residual upper", residual_bound, flush=True)
    print("E7-1 Gram correction upper", correction, "baseline 1", flush=True)
    print("E7-1 SVD and bound seconds", time.time() - start, flush=True)
    assert correction < 1
    print(
        "E7-1 PASS: positive localized identity on the exact E7-K0--E7-K3 sources",
        flush=True,
    )
    print("E7-1 sharp E3 value", E_star, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    with ExitStack() as stack:
        if args.work is None:
            args.work = Path(
                stack.enter_context(
                    tempfile.TemporaryDirectory(prefix="qi-seven-entropy-")
                )
            )
        args.work.mkdir(parents=True, exist_ok=True)
        start = time.time()
        check_probe_inputs()
        assert 4192 * x - 296 < 0
        ids = list(range(28)) + [3400, 3401, 3402, 3423] if args.pilot else list(range(3424))
        cached = sum(complete_checkpoint(args.work / f"{i}.npz") for i in ids) if args.resume else 0
        print("E7-1 trusted cached evaluations", cached, "/", len(ids), flush=True)
        if cached < len(ids):
            initialize(str(args.work), args.resume)
            # Immutable exact models and input arrays are shared by fork.
            with mp.get_context("fork").Pool(args.workers) as pool:
                for count, i in enumerate(pool.imap_unordered(evaluate, ids), 1):
                    if count % 32 == 0 or count == len(ids):
                        print("E7-1 evaluated", count, "/", len(ids), "elapsed", time.time() - start, flush=True)
        if args.pilot:
            print("E7-1 PILOT ONLY; full matrix identity not checked", flush=True)
        else:
            verify_repair(args.work, 3424)


if __name__ == "__main__":
    main()
