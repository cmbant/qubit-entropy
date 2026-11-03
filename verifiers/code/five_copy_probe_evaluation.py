"""Real direct-probe evaluation with forward-error bounds.

The certificate note defines the error budgets. All polynomial coefficients
are rational before correctly rounded conversion to binary64.
"""

import math, itertools
import numpy as np
import sympy as sp
import flint
from numba import njit
import verify_five_copy_site_algebra as s5
from five_copy_physical_faces import integer_matrix
from five_copy_probe_models import mons


def mul(p, q):
    return tuple(p[q[i]] for i in range(5))


def cosets(flip):
    H = (
        s5.PERMS
        if flip == 0
        else (
            [p for p in s5.PERMS if p[4] == 4]
            if flip == 1
            else [p for p in s5.PERMS if set(p[3:]) == {3, 4}]
        )
    )
    seen = set()
    left = []
    for p in s5.PERMS:
        if p in seen:
            continue
        left.append(s5.PINDEX[p])
        seen.update(mul(h, p) for h in H)
    right = [s5.PINDEX[tuple(np.argsort(s5.PERMS[i]))] for i in left]
    L = [s5.PERMS[i] for i in left]
    R = [s5.PERMS[i] for i in right]
    lc = {mul(h, p): i for i, p in enumerate(L) for h in H}
    rc = {mul(p, h): i for i, p in enumerate(R) for h in H}
    pairs = set(itertools.product(range(len(L)), repeat=2))
    orbs = []
    while pairs:
        a, b = min(pairs)
        o = {(lc[mul(L[a], q)], rc[mul(q, R[b])]) for q in [(0, 1, 2, 3, 4), (1, 0, 2, 3, 4)]}
        orbs.append((a, b, len(o)))
        pairs -= o
    return left, right, orbs


def local_maps(meta):
    C4 = sp.Matrix(meta["local"]["4"]["C"])
    C5 = sp.Matrix(meta["local"]["5"]["C"])
    Conj = sp.diag(
        sp.ones(1), sp.kronecker_product(C4, C4.inv().T), sp.kronecker_product(C5, C5.inv().T)
    )
    cq = flint.fmpq_mat([[flint.fmpq(str(v)) for v in row] for row in Conj.tolist()])
    out = {}
    for flip in [0, 1, 2]:
        phi = sp.eye(42) if flip == 0 else s5.flip_map((4,) if flip == 1 else (3, 4))
        pq = flint.fmpq_mat([[flint.fmpq(str(v)) for v in row] for row in phi.tolist()])
        M = cq * pq
        den = math.lcm(*[int(v.denom()) for v in M.entries()])
        MI = flint.fmpz_mat(42, 42, [int(v * den) for v in M.entries()])
        L, R, O = cosets(flip)
        mats = {}
        for a in range(len(L)):
            for b in range(len(R)):
                TT = np.zeros((42, 42), np.int64)
                p = 0
                for m, d in enumerate([1, 4, 5]):
                    TT[p : p + d * d, p : p + d * d] = np.kron(
                        np.array(s5.RHO[m][L[a]], np.int64), np.array(s5.RHO[m][R[b]], np.int64).T
                    )
                    p += d * d
                KK = MI * integer_matrix(TT)
                K = np.array([[int(KK[i, j]) for j in range(42)] for i in range(42)], np.int64)
                mats[a, b] = K
        out[flip] = dict(den=den, n=len(L), orbits=O, mats=mats)
        print(
            "exact local probe maps",
            flip,
            "cosets",
            len(L),
            "C2 terms",
            len(O),
            "denominator",
            den,
            flush=True,
        )
    return out


DATA = []
for n in range(8):
    mm = mons(5, n)
    if not n:
        DATA.append(None)
        continue
    pp = mons(5, n - 1)
    idx = {m: i for i, m in enumerate(mm)}
    ip = {m: i for i, m in enumerate(pp)}
    parents = []
    letters = []
    children = np.empty((len(pp), 5), np.int64)
    for m in mm:
        j = next(i for i, v in enumerate(m) if v)
        p = list(m)
        p[j] -= 1
        parents.append(ip[tuple(p)])
        letters.append(j)
    for j, m in enumerate(pp):
        for i in range(5):
            p = list(m)
            p[i] += 1
            children[j, i] = idx[tuple(p)]
    DATA.append((np.array(parents), np.array(letters), children))


@njit
def step_abs(h, prev, par, let, ch):
    P = len(h)
    N = len(par)
    M = prev.shape[1]
    out = np.zeros((P, N, N))
    for p in range(P):
        for c in range(N):
            pc = par[c]
            j = let[c]
            for r in range(M):
                x = prev[p, r, pc]
                for i in range(5):
                    out[p, ch[r, i], c] += h[p, i, j] * x
    return out


def check_range(a):
    z = np.r_[abs(a.real).ravel(), abs(a.imag).ravel()] if np.iscomplexobj(a) else abs(a).ravel()
    assert np.isfinite(z).all()
    z = z[z != 0]
    assert not len(z) or (z.min() > 2.0**-400 and z.max() < 2.0**400)


def pairwise(values):
    while len(values) > 1:
        values = [
            values[i] + values[i + 1] if i + 1 < len(values) else values[i]
            for i in range(0, len(values), 2)
        ]
    return values[0]


def blocked_matmul(A, B, chunk):
    inner = A.shape[-1]
    return pairwise(
        [A[..., lo : lo + chunk] @ B[..., lo : lo + chunk, :] for lo in range(0, inner, chunk)]
    )


def act(M, X, reps):
    P, D, k = X.shape
    for i, fac in enumerate(M["fac"]):
        A = math.prod(M["dims"][:i])
        Di = M["dims"][i]
        B = math.prod(M["dims"][i + 1 :]) * k
        T = X.reshape(P, A, Di, B).transpose(0, 2, 1, 3).reshape(P, Di, A * B)
        Y = reps[fac] @ T
        X = Y.reshape(P, Di, A, B).transpose(0, 2, 1, 3).reshape(P, D, k)
        check_range(X)
    return X


def matmul_real(A, B, chunk):
    return pairwise(
        [
            A[..., lo : lo + chunk] @ B[..., lo : lo + chunk, :]
            for lo in range(0, A.shape[-1], chunk)
        ]
    )


def act_real(M, X, reps, chunk=32):
    P, D, k = X.shape
    for i, fac in enumerate(M["fac"]):
        A = math.prod(M["dims"][:i])
        Di = M["dims"][i]
        B = math.prod(M["dims"][i + 1 :]) * k
        T = X.reshape(P, A, Di, B).transpose(0, 2, 1, 3).reshape(P, Di, A * B)
        Y = matmul_real(reps[fac], T, chunk)
        X = Y.reshape(P, Di, A, B).transpose(0, 2, 1, 3).reshape(P, D, k)
        check_range(X)
    return X


def left_product(VL, X, chunk=128):
    return matmul_real(VL.T, X, chunk)


def trace_product(VL, X, chunk=32):
    D = VL.shape[0]
    parts = [np.sum(VL[None, lo : lo + chunk] * X[:, lo : lo + chunk], axis=1) for lo in range(0, D, chunk)]
    diags = pairwise(parts)
    return pairwise([diags[:, j] for j in range(diags.shape[1])])


def powers_real(h):
    cp = {}
    ab = {}
    prev = np.ones((len(h), 1, 1))
    pa = prev.copy()
    ha = abs(h)
    check_range(h)
    for n in range(1, 8):
        prev = step_abs(h, prev, *DATA[n])
        pa = step_abs(ha, pa, *DATA[n])
        cp[5, n] = prev
        ab[5, n] = pa
        check_range(prev)
        check_range(pa)
    return cp, ab


def symbol(models, maps, input_num, input_den, kind, batch=8, chunks=(32, 128, 32)):
    actor_chunk, product_chunk, trace_chunk = chunks
    assert all(isinstance(c, int) and 1 <= c <= 1024 for c in chunks)
    outputs = [np.zeros((M["repair_rank"], M["VR"].shape[1])) for M in models]
    bounds = [np.zeros_like(x.real) for x in outputs]
    traces = np.zeros(len(models))
    trace_abs = np.zeros(len(models))
    tradius = np.zeros(len(models))
    for flip in [0, 1, 2]:
        dd = maps[flip]
        N = dd["n"]
        pairs = dd["orbits"] if kind == 4 else [(a, b, 1) for a in range(N) for b in range(N)]
        mm = [(i, M) for i, M in enumerate(models) if M["flip"] == flip]
        raw = []
        weights = []
        for a, b, w in pairs:
            K = dd["mats"][a, b]
            assert 42 * int(abs(K).max()) * int(abs(input_num).max()) < 2**53
            rr = K @ input_num[0]
            ii = K @ input_num[1]
            assert 0 < input_den * dd["den"] < 2**53
            den = float(input_den * dd["den"])
            assert not np.any(ii)
            raw.append(rr / den)
            weights.append(w)
        for lo in range(0, len(raw), batch):
            z = np.array(raw[lo : lo + batch])
            P = len(z)
            rp, ra = powers_real(z[:, 17:].reshape(P, 5, 5))
            hh = z[:, 1:17].reshape(P, 4, 4)
            rp[4, 1] = hh
            ra[4, 1] = abs(hh)
            ws = np.array(weights[lo : lo + P])
            for index, M in mm:
                D, k = M["VR"].shape
                r = M["repair_rank"]
                X = act_real(M, np.broadcast_to(M["VR"][None], (P, D, k)).copy(), rp, actor_chunk)
                Y = act_real(M, np.broadcast_to(M["VRU"][None], (P, D, r)).copy(), rp, actor_chunk)
                AX = act(M, np.broadcast_to(abs(M["VR"])[None], (P, D, k)).copy(), ra)
                AY = act(M, np.broadcast_to(abs(M["VRU"])[None], (P, D, r)).copy(), ra)
                Z = left_product(M["VLU"], X, product_chunk) + left_product(M["VL"], Y, product_chunk).transpose(0, 2, 1)
                AZ = abs(M["VLU"]).T @ AX + (abs(M["VL"]).T @ AY).transpose(0, 2, 1)
                tv = trace_product(M["VL"], X, trace_chunk)
                at = np.einsum("ij,pij->p", abs(M["VL"]), AX)
                coef = M["coef"] / (N * N)
                outputs[index] += np.einsum("p,pij->ij", ws, Z) * coef
                bounds[index] += np.einsum("p,pij->ij", ws, AZ) * coef
                traces[index] += np.dot(ws, tv) * coef
                trace_abs[index] += np.dot(ws, at) * coef
        for index, M in mm:
            D, k = M["VR"].shape
            actor = sum(min(di, actor_chunk) + math.ceil(math.log2(math.ceil(di / actor_chunk))) for di in M["dims"])
            final = min(D, product_chunk) + math.ceil(math.log2(math.ceil(D / product_chunk)))
            diag = (
                min(D, trace_chunk) + math.ceil(math.log2(math.ceil(D / trace_chunk))) + math.ceil(math.log2(k)) + 1
            )
            degree = sum(n for d, n in M["fac"])
            summation = batch + math.ceil(len(pairs) / batch) + 16
            ops = actor + final + 6 * degree + summation
            tops = actor + diag + 6 * degree + summation
            absops = D * k + sum(M["dims"]) + 6 * degree + 2 * N * N + 32
            u = 2**-53
            g = ops * u / (1 - ops * u)
            tg = tops * u / (1 - tops * u)
            ag = absops * u / (1 - absops * u)
            infl = 1 / (1 - ag)
            bounds[index] *= g * infl * (1 + 2**-20)
            tradius[index] = tg * infl * (1 + 2**-20) * (1 + 2**-20) * trace_abs[index]
    return (
        np.concatenate([z.ravel() for z in outputs]),
        np.concatenate([b.ravel() for b in bounds]),
        pairwise(list(traces)),
        sum(tradius),
    )
