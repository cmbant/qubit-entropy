"""E7-1 physical probes with Arb input radii and norm companions.

The signed contractions and their error depths are the direct-probe ones.
For the vector n(V) of row Euclidean norms, n(T V) <= |T| n(V).
Minkowski and Cauchy therefore bound both Frobenius errors and traces
using one positive column per tensor action. No sampling enters.
"""
import math
import itertools
import verify_five_copy_site_algebra as s5
import numpy as np
from five_copy_probe_evaluation import (
    DATA, step_abs, pairwise, matmul_real, local_maps, mul, cosets, check_range,
)

UROUND = 2.0**-53
INFLATE = 1 + 2.0**-20
CHUNKS = (128, 256, 128)


def row_norms(V):
    # The squared entries are normal: 2^-400 <= |V_ij| < 2^20.
    return np.linalg.norm(V, axis=1) * INFLATE


def prepare(models):
    for M in models:
        D, k = M["VR"].shape
        assert D < 2**13 and k < 2**8 and 0 < M["coef"] < 2**15
        assert sum(n for d, n in M["fac"]) == 7
        assert all(d == 5 or (d == 4 and n in (1, 2)) for d, n in M["fac"])
        for key in ("VR", "VL", "VRU", "VLU"):
            a = abs(M[key])
            assert np.isfinite(a).all() and (not a.size or a.max() < 2**20)
            assert not np.any((a != 0) & (a < 2.0**-400))
        # Absorb Arb input errors into each bilinear positive companion.
        # All signed path depths are >=64, so gamma_m >=64u covers both
        # rounding and every term containing an input-radius factor.
        envelope = {}
        for key in ("VR", "VL", "VRU", "VLU"):
            radius = M["radii"][key]
            assert radius.shape == M[key].shape and np.all(radius >= 0)
            envelope[key] = (abs(M[key]) + radius / (64 * UROUND)) * (1 + 2.0**-40)
        M["envelope_R"] = row_norms(envelope["VR"])[:, None]
        M["envelope_U"] = row_norms(envelope["VRU"])[:, None]
        M["abs_VL"] = envelope["VL"]
        M["abs_VLU"] = envelope["VLU"]
        M["trace_envelope"] = row_norms(envelope["VL"])
        # A trace needs the multiplier once, rather than its action on k
        # separate columns. The small faces remain cheaper in factored form.
        M["density"] = None
        if (k - M["repair_rank"]) * sum(M["dims"]) > D:
            dims = M["dims"]
            nf = len(dims)
            order = sorted(range(nf), key=lambda i: -dims[i])
            axes = [a for i in order[1:] + order[:1] for a in (nf + i, i)]
            C = M["VR"] @ M["VL"].T
            M["density"] = C.reshape(dims + dims).transpose(axes).reshape(-1, dims[order[0]]**2).copy()
            M["density_factors"] = [M["fac"][i] for i in order]


# Share the independently cross-checked tensor kernels and trace contraction.
from five_copy_compact_evaluation import powers, act, density_trace


def two_f(h, wedge, positive=False):
    pairs = list(itertools.combinations(range(4), 2) if wedge
                 else itertools.combinations_with_replacement(range(4), 2))
    return np.array([[h[:, c, a] * h[:, d, b] +
                      ((1 if positive or not wedge else -1) * h[:, d, a] * h[:, c, b]
                       if wedge or c != d else 0)
                      for a, b in pairs] for c, d in pairs]).transpose(2, 0, 1)


def symbol(models, maps, input_num, input_den, kind, batch=8, chunks=CHUNKS):
    actor_chunk, product_chunk, trace_chunk = chunks
    assert all(isinstance(c, int) and 1 <= c <= 1024 for c in chunks)
    assert input_num.shape == (2, 42) and not np.any(input_num[1])
    assert input_den > 0 and kind in (4, 5)
    maximum = int(abs(input_num).max())
    outputs = [np.zeros((M["repair_rank"], M["VR"].shape[1])) for M in models]
    left = [np.zeros(M["repair_rank"]) for M in models]
    right = [np.zeros(M["VR"].shape[1]) for M in models]
    bounds = np.zeros(len(models))
    traces, errors = np.zeros(len(models)), np.zeros(len(models))
    for flip in (0, 1, 2):
        dd = maps[flip]
        N = dd["n"]
        pairs = dd["orbits"] if kind == 4 else [(a, b, 1) for a in range(N) for b in range(N)]
        mm = [(i, M) for i, M in enumerate(models) if M["flip"] == flip]
        if not mm:
            continue
        raw, weights = [], []
        for a, b, w in pairs:
            K = dd["mats"][a, b]
            # Exact integer guards also give the uniform |h_ij| < 2^10 bound.
            assert 42 * int(abs(K).max()) * maximum < 2**53
            den = input_den * dd["den"]
            assert 0 < den < 2**53
            assert int(abs(K).sum(axis=1).max()) * maximum < 2**10 * den
            raw.append((K @ input_num[0]) / float(den))
            weights.append(w)
        trace_upper = np.zeros(len(models))
        for lo in range(0, len(raw), batch):
            z = np.array(raw[lo:lo + batch])
            P = len(z)
            rp, ra = powers(z[:, 17:].reshape(P, 5, 5))
            rp[4, 1] = z[:, 1:17].reshape(P, 4, 4)
            ra[4, 1] = abs(rp[4, 1])
            # The symmetric and exterior F^2 modules have the same (4,2)
            # key, so choose their representation separately for each face.
            rpt = {fac: R.transpose(0, 2, 1) for fac, R in rp.items()}
            ws = np.array(weights[lo:lo + P])
            for index, M in mm:
                D, k = M["VR"].shape
                r = M["repair_rank"]
                if (4, 2) in M["fac"]:
                    hh = rp[4, 1]
                    rp[4, 2] = two_f(hh, M["wedge"])
                    ra[4, 2] = two_f(abs(hh), M["wedge"], positive=True)
                    rpt[4, 2] = rp[4, 2].transpose(0, 2, 1)
                if M["density"] is None:
                    X = act(M, np.broadcast_to(M["VR"], (P, D, k)).copy(), rp, actor_chunk)
                    Z = matmul_real(M["VLU"].T, X, product_chunk)
                    diag = pairwise([
                        np.sum(M["VL"][None, i:i + trace_chunk] * X[:, i:i + trace_chunk], axis=1)
                        for i in range(0, D, trace_chunk)
                    ])
                    tv = pairwise([diag[:, j] for j in range(k)])
                else:
                    if r:
                        X = act(M, np.broadcast_to(M["VLU"], (P, D, r)).copy(), rpt, actor_chunk)
                        Z = matmul_real(M["VR"].T, X, product_chunk).transpose(0, 2, 1)
                    tv = density_trace(M, rp, trace_chunk)
                AX = act(M, np.broadcast_to(M["envelope_R"], (P, D, 1)).copy(), ra)
                if r:
                    Y = act(M, np.broadcast_to(M["VRU"], (P, D, r)).copy(), rp, actor_chunk)
                    AY = act(M, np.broadcast_to(M["envelope_U"], (P, D, 1)).copy(), ra)
                    Z += matmul_real(M["VL"].T, Y, product_chunk).transpose(0, 2, 1)
                    AZ = (M["abs_VLU"].T @ AX)[:, :, 0]
                    BZ = (M["abs_VL"].T @ AY)[:, :, 0]
                at = AX[:, :, 0] @ M["trace_envelope"]
                coef = M["coef"] / (N * N)
                if r:
                    outputs[index] += np.einsum("p,pij->ij", ws, Z) * coef
                    left[index] += (ws @ AZ) * coef
                    right[index] += (ws @ BZ) * coef
                traces[index] += np.dot(ws, tv) * coef
                trace_upper[index] += np.dot(ws, at) * coef
        for index, M in mm:
            D, k = M["VR"].shape
            depth = lambda d, c: min(d, c) + math.ceil(math.log2(math.ceil(d / c)))
            actor = sum(depth(d, actor_chunk) for d in M["dims"])
            summation = batch + math.ceil(len(pairs) / batch) + 16
            ops = actor + depth(D, product_chunk) + 42 + summation
            tops = actor + depth(D, trace_chunk) + math.ceil(math.log2(k)) + 43 + summation
            if M["density"] is not None:
                tops = k + 2 + sum(depth(d*d, trace_chunk) for d in M["dims"]) + 42 + summation
            assert ops >= 64 and tops >= 64
            absops = D * k + sum(M["dims"]) + 42 + 2 * N * N + 64
            gamma = lambda n: n * UROUND / (1 - n * UROUND)
            inflation = INFLATE / (1 - gamma(absops))
            companion = (np.linalg.norm(left[index]) + np.linalg.norm(right[index])) * INFLATE
            bounds[index] = gamma(ops) * inflation * companion
            errors[index] = gamma(tops) * inflation * INFLATE * trace_upper[index]
            # The entry bounds above put all mathematical intermediates below
            # 2^200. Propagated subnormal errors are below 2^-500; include them
            # explicitly instead of scanning every large intermediate array.
            bounds[index] += 2.0**-500
            errors[index] += 2.0**-500
    row = np.concatenate([a.ravel() for a in outputs])
    radius = np.array(np.linalg.norm(bounds) * INFLATE)
    assert np.isfinite(row).all() and np.isfinite(radius).all()
    assert np.isfinite(traces).all() and np.isfinite(errors).all()
    return row, radius, pairwise(list(traces)), sum(errors)
