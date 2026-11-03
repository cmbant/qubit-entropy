"""Independent exact-data cross-check of the FP5-1 probe evaluator on non-scalar probes.

FP5-2 only tests the evaluator on scalar probes (10 of 12 faces vanish).  Here, for several
genuinely non-scalar probes G (stored probes of both kinds and seeded random commutant elements),
every one of the twelve face functionals

    tr[W_b^T Gamma_b(G) W_b]   and the 8965 thin coefficients (V_b^T-repair directions)

is recomputed WITHOUT the evaluator's code path and compared with it:
  * local transport: phi_j(U_a G U_b) is formed physically as a 32x32 matrix from bit-permutation
    operators and the spin-flip partial transpose (own implementation, own LDL Gram transport),
    then compared with the evaluator's integer map K (T1);
  * the polynomial models W_b (frames, Gram transport, Sym^n) are rebuilt from the stored JSON
    data in 64-bit-mantissa floats with an independent Sym^n recursion (T0);
  * all N^2 coset pairs are summed (no (01)-orbit reduction), at longdouble precision;
  * T0 also checks: H-invariance of every W_b, Young-module membership (Pieri maps), coset
    partitions, and the multiplicity weights coef = placements * spin dimensions * dim Specht.
A wrong-convention negative control (transposed local blocks) must be detected on the same data.
Not checked here: that the stored kernels W_b are the physical moment kernels (PK5-*, PK5-4).
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import itertools, json, math, multiprocessing as mp, pickle, sys, time
from fractions import Fraction as Fr
from pathlib import Path
import flint
import numpy as np
import verify_five_copy_site_algebra as s5
import five_copy_compact_evaluation as ev
from five_copy_probe_models import build
from five_copy_physical_faces import integer_matrix, models as fourmodels
from five_copy_allv_young_models import invariant_basis
from five_copy_unflipped_oneflip_faces import models as othermodels

LD = np.longdouble
DATA = Path(__file__).resolve().parent.parent / "data"
COPIES = {0: (), 1: (4,), 2: (3, 4)}
HGRP = {0: lambda p: True, 1: lambda p: p[4] == 4, 2: lambda p: set(p[3:]) == {3, 4}}


def q2ld(q):  # flint fmpq -> longdouble
    return LD(int(q.p)) / LD(int(q.q))


def int2ld(v, e=0):  # python int / 2^e, ~1e-19 relative
    s = -1 if v < 0 else 1
    v = abs(int(v)); sh = max(v.bit_length() - 62, 0)
    return s * LD(v >> sh) * LD(2.0) ** (sh - e)


def gram_transport():  # g in V-basis -> C g C^-1 in a basis with diagonal Gram d (G = L d L^T, C = L^T)
    out = {}
    for m in (1, 2):
        G = s5.V[m].T * s5.V[m]; n = G.rows
        L = [[Fr(0)] * n for _ in range(n)]; d = [Fr(0)] * n
        for j in range(n):
            d[j] = Fr(int(G[j, j].p), int(G[j, j].q)) - sum(L[j][k] ** 2 * d[k] for k in range(j))
            L[j][j] = Fr(1)
            for i in range(j + 1, n):
                L[i][j] = (Fr(int(G[i, j].p), int(G[i, j].q)) - sum(L[i][k] * L[j][k] * d[k] for k in range(j))) / d[j]
        out[m] = (flint.fmpq_mat(n, n, [flint.fmpq(L[j][i].numerator, L[j][i].denominator) for i in range(n) for j in range(n)]), d)
    return out


def fq(M):
    return flint.fmpq_mat(M.rows, M.cols, [flint.fmpq(int(v.p), int(v.q)) for v in M])


V32 = {m: fq(s5.V32[m]) for m in range(3)}
L32 = {m: fq(s5.L32[m]) for m in range(3)}


def perm_op(p):  # basis state: bit i of copy i moves to position p[i]
    M = np.zeros((32, 32), dtype=object)
    for x in range(32):
        y = [0] * 5
        for i in range(5): y[p[i]] = (x >> (4 - i)) & 1
        M[sum(b << (4 - i) for i, b in enumerate(y)), x] = 1
    return M


def flip_phys(A, copies):  # (eps^T A eps)^{T_J} on the copies J
    eps = np.array([[0, 1], [-1, 0]], dtype=object)
    T = A.reshape((2,) * 10)
    for c in copies:
        T = np.moveaxis(np.tensordot(eps, T, axes=([0], [c])), 0, c)
        T = np.moveaxis(np.tensordot(T, eps, axes=([5 + c], [0])), -1, 5 + c)
    ax = list(range(10))
    for c in copies: ax[c], ax[5 + c] = ax[5 + c], ax[c]
    return T.transpose(ax).reshape(32, 32)


def phys_blocks(xnum, den, Lp, Rp, flip, CT):
    """transported (g4, g5) of phi_flip(U_L G U_R), G = sum_c x_c E_c / (120 den); exact -> longdouble."""
    G = sum(int(x) * np.asarray(E, dtype=object) for x, E in zip(xnum, s5.EMB_NUM))
    A = perm_op(Lp).dot(G).dot(perm_op(Rp))
    if COPIES[flip]: A = flip_phys(A, COPIES[flip])
    Af = flint.fmpq_mat(32, 32, [flint.fmpq(int(v), 1) for v in A.ravel()])
    sc = flint.fmpq(1, 120 * den)
    out, exact = {}, []
    for m in range(3):
        B = L32[m] * Af * V32[m]
        if m: B = CT[m][0] * B * CT[m][0].inv()
        B = B * sc
        out[m] = B; exact += [(int(B[i, j].p), int(B[i, j].q)) for i in range(B.nrows()) for j in range(B.ncols())]
    ld = lambda B: np.array([[q2ld(B[i, j]) for j in range(B.ncols())] for i in range(B.nrows())], dtype=LD)
    return ld(out[1]), ld(out[2]), exact


_MON = {}
def mons(n):
    if n not in _MON:
        ms = [tuple(c.count(i) for i in range(5)) for c in itertools.combinations_with_replacement(range(5), n)]
        _MON[n] = (ms, {m: i for i, m in enumerate(ms)})
    return _MON[n]


def sympow(h, n):  # Sym^n(h), column = exponent vector; independent last-letter recursion
    if h.shape[0] == 4:
        assert n == 1; return h.copy()
    if n == 0: return np.ones((1, 1), dtype=h.dtype)
    ms, ix = mons(n); pms, pix = mons(n - 1)
    ch = np.array([[ix[tuple(a[k] + (k == i) for k in range(5))] for i in range(5)] for a in pms])
    prev = sympow(h, n - 1); S = np.zeros((len(ms), len(ms)), dtype=h.dtype)
    for c, m in enumerate(ms):
        j = max(k for k in range(5) if m[k]); pm = list(m); pm[j] -= 1
        for i in range(5): S[ch[:, i], c] += h[i, j] * prev[:, pix[tuple(pm)]]
    return S


def tensor_act(Ss, dims, X):
    k = X.shape[1]; X = X.reshape(*dims, k)
    for i, S in enumerate(Ss): X = np.moveaxis(np.tensordot(S, X, axes=([1], [i])), 0, i)
    return X.reshape(-1, k)


def build_face(rec, fr, U, CT):
    K = flint.fmpz_mat(rec["kernel_columns"])
    B = flint.fmpz_mat([[int(v) for v in row] for row in rec["basis_numerator"]])
    UKB = integer_matrix(U) * K * B
    X = np.array([[int2ld(UKB[i, j], rec["basis_denominator_exponent"]) for j in range(UKB.ncols())] for i in range(UKB.nrows())], dtype=LD)
    fac = [tuple(f) for f in rec["fac"]]; Ss, hd = [], []
    for d, n in fac:
        C, dd = CT[1 if d == 4 else 2]
        Ss.append(sympow(np.array([[q2ld(C[i, j]) for j in range(d)] for i in range(d)], dtype=LD), n))
        dl = [LD(x.numerator) / LD(x.denominator) for x in dd]
        hd.append(np.array(dl, dtype=LD) if d == 4 else np.array(
            [LD(math.prod(math.factorial(a) for a in m)) / LD(math.factorial(n)) * np.prod([dl[i] ** m[i] for i in range(5)]) for m in mons(n)[0]], dtype=LD))
    dims = [len(h) for h in hd]; W = tensor_act(Ss, dims, X)
    H = hd[0]
    for h in hd[1:]: H = np.outer(H, h).ravel()
    r = fr["rank"]; Q = np.eye(W.shape[1], r, dtype=LD)
    for v in fr["vectors"]:
        v = np.array([int2ld(x) for x in v], dtype=LD); Q = Q - 2 * np.outer(v, v @ Q) / (v @ v)
    assert np.max(np.abs(Q.T @ Q - np.eye(r, dtype=LD))) < 1e-17
    return dict(tag=rec["tag"], flip=rec["flip"], fac=fac, dims=dims, W=W, WV=W @ Q, H=H, coef=int(rec["coefficient"]), r=r)


def functional(F, blocks, transpose=False):
    """(trace, thin) of coef * average over the given pairs of <W|rho(g)|W> and the symmetrised repair forms."""
    k, r = F["W"].shape[1], F["r"]; both = np.hstack([F["W"], F["WV"]])
    HW, HV = F["H"][:, None] * F["W"], F["H"][:, None] * F["WV"]
    tr, Z = LD(0), np.zeros((r, k), dtype=LD)
    for g4, g5 in blocks:
        if transpose: g4, g5 = g4.T, g5.T
        Xb = tensor_act([sympow(g4 if d == 4 else g5, n) for d, n in F["fac"]], F["dims"], both)
        tr += np.sum(HW * Xb[:, :k]); Z += HV.T @ Xb[:, :k] + (HW.T @ Xb[:, k:]).T
    s = LD(F["coef"]) / LD(len(blocks))
    return tr * s, Z * s


def specht_dim(lam):  # hook length formula
    n = sum(lam); cols = [sum(1 for r in lam if r > c) for c in range(lam[0])]
    hooks = math.prod(lam[i] - j + cols[j] - i - 1 for i in range(len(lam)) for j in range(lam[i]))
    return math.factorial(n) // hooks


def pieri(k, l, polarize):  # monomial-basis Pieri maps (own construction)
    ms_k, ix_k = mons(k); e = [tuple(int(i == j) for i in range(5)) for j in range(5)]
    add = lambda a, b: tuple(x + y for x, y in zip(a, b))
    if not polarize:  # Sym^k x V -> Sym^{k+1}
        hi = mons(k + 1)[1]; E = np.zeros((len(hi), len(ms_k) * 5), dtype=LD)
        for i, a in enumerate(ms_k):
            for j in range(5): E[hi[add(a, e[j])], i * 5 + j] += 1
        return E
    ms_l = mons(l)[0]; hi = mons(k + 1)[1]; E = np.zeros((len(hi) * 5, len(ms_k) * len(ms_l)), dtype=LD)
    for i, a in enumerate(ms_k):  # delta: y^a x y^b -> sum_u b_u y^{a+e_u} x y^{b-e_u}  (l = 2, result in Sym^1 = V)
        for j, b in enumerate(ms_l):
            for u in range(5):
                if b[u]:
                    v = list(b); v[u] -= 1; E[hi[add(a, e[u])] * 5 + v.index(1), i * len(ms_l) + j] += b[u]
    return E


def group_checks(faces, CT, cos):
    ok = True
    # coset partitions: left reps of H\S5 and right reps of S5/H
    mul = ev.mul
    for f in (0, 1, 2):
        H = [p for p in s5.PERMS if HGRP[f](p)]
        Li, Ri = cos[f]; L = [s5.PERMS[i] for i in Li]; R = [s5.PERMS[i] for i in Ri]
        left = {mul(h, p) for p in L for h in H}; right = {mul(p, h) for p in R for h in H}
        good = len(left) == 120 == len(L) * len(H) and len(right) == 120 == len(R) * len(H)
        print(f"T0 flip {f}: |H|={len(H)}, {len(L)} left and {len(R)} right coset reps partition S5 {'PASS' if good else 'FAIL'}")
        ok &= good
    for F in faces:
        has_f = any(d == 4 for d, n in F["fac"]); lam = tuple(int(c) for c in F["tag"].split("-")[-1])
        exp = (7 * 4 * 2 ** 6 if has_f else 2 ** 7) * specht_dim(lam); good = exp == F["coef"]
        ok &= good
        print(f"T0 face {F['flip']}:{F['tag']}: coef {F['coef']} = placements*spin*dim Specht {lam} -> {'PASS' if good else 'FAIL'}")
    gens = {0: [(1, 0, 2, 3, 4), (1, 2, 3, 4, 0)], 1: [(1, 0, 2, 3, 4), (1, 2, 3, 0, 4)], 2: [(1, 0, 2, 3, 4), (1, 2, 0, 3, 4), (0, 1, 2, 4, 3)]}
    for F in faces:
        worst = 0.0
        for p in gens[F["flip"]]:
            idx = s5.PINDEX[p]; gs = {}
            for m in (1, 2):
                R = flint.fmpq_mat(s5.RHO[m][idx].rows, s5.RHO[m][idx].cols, [int(v) for v in s5.RHO[m][idx]])
                C = CT[m][0]; B = C * R * C.inv()
                gs[m] = np.array([[q2ld(B[i, j]) for j in range(B.ncols())] for i in range(B.nrows())], dtype=LD)
            X = tensor_act([sympow(gs[1 if d == 4 else 2], n) for d, n in F["fac"]], F["dims"], F["W"])
            worst = max(worst, float(np.max(np.abs(X - F["W"])) / np.max(np.abs(F["W"]))))
        young = ""
        if F["tag"].split("-")[-1] in ("61", "52", "51", "42"):
            kk, ll = F["fac"][-2][1], F["fac"][-1][1]
            E = pieri(kk, ll, ll == 2); lead = int(np.prod(F["dims"][:-2])) if len(F["dims"]) > 2 else 1
            Wr = F["W"].reshape(lead, F["dims"][-2] * F["dims"][-1], -1)
            ey = max(float(np.max(np.abs(E @ Wr[f]))) for f in range(lead)) / float(np.max(np.abs(F["W"])))
            young = f"; Pieri map annihilates W: {ey:.1e}"; worst = worst if ey < 1e-15 else 1.0
        good = worst < 1e-15; ok &= good
        print(f"T0 face {F['flip']}:{F['tag']}: H-invariance defect {worst:.1e}{young} -> {'PASS' if good else 'FAIL'}")
    return ok


G = {}  # shared (fork) state


def probe_job(job):
    name, num, kind = job
    den = G["den"]; num = np.asarray(num, dtype=np.int64); res = dict(name=name, kind=kind)
    bad, blocks = 0, {}
    for f in (0, 1, 2):  # T1: evaluator's integer local maps against the physical construction, exact
        L, R = G["cos"][f]; dd = G["maps"][f]; blocks[f] = []
        for a, ia in enumerate(L):
            for b, ib in enumerate(R):
                g4, g5, exact = phys_blocks(num[0], den, s5.PERMS[ia], s5.PERMS[ib], f, G["CT"])
                Kx = dd["mats"][a, b] @ num[0]
                bad += sum(Fr(int(Kx[c]), den * dd["den"]) != Fr(*exact[c]) for c in range(42))
                blocks[f].append((g4, g5))
    res["T1_bad_entries"] = bad
    rows, trs, worst_t, worst_z = [], LD(0), 0.0, 0.0
    for M, F in zip(G["models"], G["faces"]):
        row, rad, tev, terr = ev.symbol([M], G["maps"], num, den, kind)
        tr, Z = functional(F, blocks[F["flip"]]); Zr = Z.astype(float).ravel()
        worst_t = max(worst_t, abs(float(tr) - tev) / (terr * (1 + 1e-6) + 1e-14 * abs(float(tr)) + 1e-300))
        worst_z = max(worst_z, float(np.linalg.norm(Zr - row) / (rad * (1 + 1e-6) + 1e-14 * np.linalg.norm(Zr) + 1e-300)))
        rows.append(Zr); trs += tr
        res[f"{F['flip']}:{F['tag']}"] = (float(tr), tev, float(np.max(np.abs(Zr))))
    row, rad, tev, terr = ev.symbol(G["models"], G["maps"], num, den, kind)
    ref = np.concatenate(rows)
    res.update(len=len(row), worst_trace_ratio=worst_t, worst_thin_ratio=worst_z, total_trace=(float(trs), tev),
               total_ok=bool(abs(float(trs) - tev) <= terr + 1e-13 * abs(tev) and np.linalg.norm(ref - row) <= rad + 1e-13 * np.linalg.norm(ref)))
    # negative control: the transposed local-block convention must be refuted by non-scalar probes
    F = G["faces"][0]; tr2, _ = functional(F, blocks[F["flip"]], transpose=True)
    res["neg_control"] = (float(tr2), res[f"{F['flip']}:{F['tag']}"][0])
    return res


def invariance_samples(faces, CT, den, num):
    rng = np.random.default_rng(7); ok = True
    for F in (f for f in faces if f["tag"] in ("0-7", "1-7", "7")):
        H = [p for p in s5.PERMS if HGRP[F["flip"]](p)]; vals = []
        for _ in range(3):
            s, t = (s5.PERMS[rng.integers(120)] for _ in range(2)); h1, h2 = (H[rng.integers(len(H))] for _ in range(2))
            v = lambda a, b: float(functional(F, [phys_blocks(num, den, a, b, F["flip"], CT)[:2]])[0])
            base = v(s, t); vals.append(max(abs(v(ev.mul(h1, s), t) - base), abs(v(s, ev.mul(t, h2)) - base), abs(v(ev.mul(h1, s), ev.mul(t, h2)) - base)) / (abs(base) + 1e-30))
        good = max(vals) < 1e-14; ok &= good
        print(f"T2 face {F['flip']}:{F['tag']}: value of phi(U_s G U_t) unchanged under s->hs, t->th' (rel {max(vals):.1e}) {'PASS' if good else 'FAIL'}")
    return ok


def main():
    t0 = time.time(); probes = np.load(DATA / "five_copy_repair_probes.npz"); den = int(probes["denominator"])
    models, meta = build(); G.update(models=models, maps=ev.local_maps(meta), den=den, CT=gram_transport())
    ev.prepare(models)
    G["cos"] = {f: ev.cosets(f)[:2] for f in (0, 1, 2)}
    from five_copy_candidate_data import load_candidate, load_frames
    frames = load_frames()
    recs = load_candidate()
    base = [m[1] for m in fourmodels()] + [invariant_basis(6, 1), invariant_basis(5, 2)] + [m[4] for m in othermodels()]
    G["faces"] = faces = [build_face(r, f, U, G["CT"]) for r, f, U in zip(recs, frames, base)]
    print(f"setup {time.time() - t0:.0f}s: 12 faces rebuilt from stored data; stored VR vs rebuilt W max rel diff",
          f"{max(np.max(np.abs(F['W'].astype(float) - M['VR'])) / np.max(np.abs(M['VR'])) for F, M in zip(faces, models)):.1e}")
    ok = group_checks(faces, G["CT"], G["cos"])
    rng = np.random.default_rng(59)
    jobs = [(f"stored#{i}", probes["inputs"][i][0], int(probes["kinds"][i])) for i in (0, 700, 1500, 2600, 3000, 3399, 3400, 3407, 3414, 3423)]
    jobs += [(f"random#{j}", rng.integers(-3000, 3001, size=(1, 42))[0].astype(np.int64), 5) for j in range(2)]
    jobs = [(n, np.stack([x, np.zeros(42, np.int64)]), k) for n, x, k in jobs]
    ok &= invariance_samples(faces, G["CT"], den, jobs[4][1][0])
    jobs = jobs[: int(os.environ.get("XC_N", len(jobs)))]  # debugging aid only; default runs all
    with mp.get_context("fork").Pool(int(os.environ.get("XC_WORKERS", "4"))) as pool: results = pool.map(probe_job, jobs, chunksize=1)
    nrow = results[0]["len"]; assert nrow == 8965
    for r in results:
        nz = sum(1 for k, v in r.items() if isinstance(v, tuple) and ":" in k and v[2] > 1e-9)
        good = (r["T1_bad_entries"] == 0 and r["worst_trace_ratio"] <= 1 and r["worst_thin_ratio"] <= 1 and r["total_ok"]); ok &= good
        print(f"{r['name']} (kind {r['kind']}): local transport exact mismatches {r['T1_bad_entries']}; all 12 face traces and {nrow} thin coefficients "
              f"inside evaluator radii (worst |d|/radius: trace {r['worst_trace_ratio']:.2e}, thin {r['worst_thin_ratio']:.2e}); {nz}/12 faces nonzero; "
              f"summed trace {r['total_trace'][0]:.12g} vs {r['total_trace'][1]:.12g} -> {'PASS' if good else 'FAIL'}")
        a, b = r["neg_control"]; refuted = abs(a - b) > 1e-6 * abs(b); ok &= refuted
        print(f"  negative control (transposed local blocks, face 2:7): {a:.6g} vs {b:.6g} -> {'REFUTED as required' if refuted else 'NOT DETECTED (FAIL)'}")
    print(f"crosscheck {'PASS' if ok else 'FAIL'}: {len(results)} non-scalar probes x 12 faces; seconds {time.time() - t0:.0f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
