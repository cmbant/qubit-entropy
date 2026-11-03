"""TT-1: exact five-copy column parity and rational kernel controls.

The additive support charge reduces the three unflipped all-V feature
contractions to 14,432 five-multisets. A diagonal local symmetry fixes u
and negates v, forcing disjoint even/odd column supports. Check this on
the three reconstructed faces and four stored IK5 arrays, then check
that the stored physical kernels annihilate both field components.
Equality uses the invariant-square lemma; no transversality test is needed.
"""

from fractions import Fraction
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import flint
import scipy.sparse as ss
import sympy as sp

import verify_five_copy_site_algebra as s5

DATA = Path(__file__).resolve().parent.parent / "data"
P = 2147483629
D = 5
CYCLE = (0, 1, 4, 5, 3, 6, 2)
assert sp.isprime(P)


def monomials(k):
    out = []
    for c in itertools.combinations_with_replacement(range(D), k):
        e = [0] * D
        for i in c:
            e[i] += 1
        out.append(tuple(e))
    return out


def select_columns(A):
    M = (np.asarray(A, dtype=np.int64) % P).copy()
    nrows, ncols = M.shape
    piv = []
    row = 0
    for col in range(ncols):
        if row >= nrows:
            break
        r = row
        while r < nrows and M[r, col] == 0:
            r += 1
        if r == nrows:
            continue
        if r != row:
            M[[row, r]] = M[[r, row]]
        inv = pow(int(M[row, col]), P - 2, P)
        M[row, col:] = (M[row, col:] * inv) % P
        nz = np.nonzero(M[row + 1 :, col])[0] + (row + 1)
        if len(nz):
            M[nz, col:] = (
                M[nz, col:] - M[nz, col : col + 1] * M[row : row + 1, col:]
            ) % P
        piv.append(col)
        row += 1
    return A[:, piv].copy()


def sym_power(R, degree):
    mons = monomials(degree)
    idx = {v: i for i, v in enumerate(mons)}
    e = [tuple(int(i == j) for i in range(D)) for j in range(D)]
    S = np.zeros((len(mons), len(mons)), dtype=np.int64)
    for col, exp in enumerate(mons):
        terms = {(0,) * D: 1}
        for j in range(D):
            for _ in range(exp[j]):
                new = {}
                for a, coeff in terms.items():
                    for i in range(D):
                        c = int(R[i, j])
                        if c:
                            b = tuple(a[h] + e[i][h] for h in range(D))
                            new[b] = new.get(b, 0) + c * coeff
                terms = new
        for a, coeff in terms.items():
            S[idx[a], col] = coeff
    return S


def symmetric_gram_numerator(n):
    G = np.asarray(s5.V[2].T * s5.V[2], dtype=np.int64)
    factorials = np.array(
        [math.prod(math.factorial(v) for v in e) for e in monomials(n)], dtype=object
    )
    H = factorials[:, None] * sym_power(G, n).astype(object)
    assert np.array_equal(H, H.T)
    return H


def apply_gram(Y, factors):
    X = Y.reshape(*[G.shape[1] for G in factors], Y.shape[1]).astype(object)
    for i, G in enumerate(factors):
        X = np.moveaxis(np.tensordot(G, X, axes=([1], [i])), 0, i)
    return X.reshape(Y.shape)


def invariant_basis_7():
    H = [k for k, p in enumerate(s5.PERMS) if set(p[3:]) == {3, 4}]
    actions = [sym_power(np.asarray(s5.RHO[2][k], dtype=np.int64), 7) for k in H]
    rng = np.random.default_rng(7)
    X0 = rng.integers(-2, 3, size=(330, 49), dtype=np.int64)
    Y0 = sum(S7 @ X0 for S7 in actions)
    U0 = select_columns(Y0)
    assert U0.shape == (330, 33)
    for S7 in actions:
        assert np.array_equal(S7 @ U0, U0)
    return U0


def invariant_basis_kl(k, l):
    mk, ml = monomials(k), monomials(l)
    ik, il = {m: i for i, m in enumerate(mk)}, {m: i for i, m in enumerate(ml)}
    e = [tuple(int(i == j) for i in range(5)) for j in range(5)]
    add = lambda *xs: tuple(sum(x[i] for x in xs) for i in range(5))
    if l == 1:
        small = monomials(k - 1)
        G = np.zeros((len(mk) * 5, len(small) * 10), np.int64)
        c = 0
        for a in small:
            for i, j in itertools.combinations(range(5), 2):
                G[ik[add(a, e[i])] * 5 + j, c] += 1
                G[ik[add(a, e[j])] * 5 + i, c] -= 1
                c += 1
        high = monomials(k + 1)
        ih = {m: i for i, m in enumerate(high)}
        E = ss.lil_matrix((len(high), len(mk) * 5), dtype=np.int64)
        for i, a in enumerate(mk):
            for j in range(5):
                E[ih[add(a, e[j])], i * 5 + j] = 1
        E = E.tocsr()
        expected = 5 * len(mk) - len(high)
    else:
        small = monomials(k - 2)
        wedges = list(itertools.combinations(range(5), 2))
        G = np.zeros((len(mk) * len(ml), len(small) * 55), np.int64)
        c = 0
        for a in small:
            for (i, j), (u, v) in itertools.combinations_with_replacement(wedges, 2):
                for (aa, bb), (cc, dd), sg in [
                    ((i, u), (j, v), 1),
                    ((i, v), (j, u), -1),
                    ((j, u), (i, v), -1),
                    ((j, v), (i, u), 1),
                ]:
                    G[
                        ik[add(a, e[aa], e[bb])] * len(ml) + il[add(e[cc], e[dd])], c
                    ] += sg
                c += 1
        high = monomials(k + 1)
        ih = {m: i for i, m in enumerate(high)}
        E = ss.lil_matrix((len(high) * 5, len(mk) * len(ml)), dtype=np.int64)
        for i, a in enumerate(mk):
            for j, b in enumerate(ml):
                for u in range(5):
                    if b[u]:
                        v = list(b)
                        v[u] -= 1
                        qq = v.index(1)
                        E[ih[add(a, e[u])] * 5 + qq, i * len(ml) + j] += b[u]
        E = E.tocsr()
        expected = len(mk) * len(ml) - len(high) * 5
    assert not np.any(E @ G)
    B = select_columns(G)
    assert B.shape[1] == expected
    H = [i for i, p in enumerate(s5.PERMS) if set(p[3:]) == {3, 4}]
    dim = 60 if l == 1 else 75
    rng = np.random.default_rng(61 if l == 1 else 52)
    X = (B @ rng.integers(-2, 3, size=(B.shape[1], dim + 16), dtype=np.int64)).reshape(
        len(mk), len(ml), dim + 16
    )
    Y = np.zeros_like(X)
    actions = []
    for i in H:
        R = np.asarray(s5.RHO[2][i], dtype=np.int64)
        S0, S1 = sym_power(R, k), sym_power(R, l)
        actions.append((S0, S1))
        Y += np.einsum("ij,jbk,ab->iak", S0, X, S1, optimize=True)
    U = select_columns(Y.reshape(len(mk) * len(ml), dim + 16))
    assert U.shape[1] == dim and (not np.any(E @ U))
    for A, C in actions:
        yy = np.einsum(
            "ij,jbk,ab->iak", A, U.reshape(len(mk), len(ml), dim), C, optimize=True
        )
        assert np.array_equal(yy.reshape(U.shape), U)
    return U


def character_dimension(k, l, group):
    total = Fraction(0)
    for i in group:
        R = s5.RHO[2][i]
        Z = R**0
        traces = [0]
        for j in range(1, 8):
            Z = Z * R
            traces.append(int(Z.trace()))
        h = [Fraction(1)]
        for j in range(1, 8):
            h.append(sum(traces[a] * h[j - a] for a in range(1, j + 1)) / j)
        total += h[k] if not l else h[k] * h[l] - h[k + 1] * h[l - 1]
    return total / len(group)


def build_unflipped_models():
    base = [
        ("7", invariant_basis_7(), 7, 0),
        ("61", invariant_basis_kl(6, 1), 6, 1),
        ("52", invariant_basis_kl(5, 2), 5, 2),
    ]
    H = [p for p in s5.PERMS if set(p[3:]) == {3, 4}]
    seen, coset_reps = set(), []
    for p in s5.PERMS:
        if p in seen:
            continue
        coset_reps.append(tuple(np.argsort(p)))
        seen.update(tuple(h[p[i]] for i in range(5)) for h in H)
    assert len(coset_reps) == 10
    group = list(range(len(s5.PERMS)))
    out = []
    for (tag, U, k, l), dim in zip(base, (4, 6, 9)):
        assert character_dimension(k, l, group) == dim
        Y = np.zeros_like(U)
        for p in coset_reps:
            R = np.array(s5.RHO[2][s5.PINDEX[p]], np.int64)
            A, B = sym_power(R, k), sym_power(R, l)
            Y += np.einsum(
                "ij,jbk,ab->iak",
                A,
                U.reshape(len(A), len(B), U.shape[1]),
                B,
                optimize=True,
            ).reshape(U.shape)
        V = select_columns(Y)
        assert V.shape[1] == dim
        for i in group:
            R = np.array(s5.RHO[2][i], np.int64)
            A, B = sym_power(R, k), sym_power(R, l)
            act = np.einsum(
                "ij,jbk,ab->iak",
                A,
                V.reshape(len(A), len(B), V.shape[1]),
                B,
                optimize=True,
            ).reshape(V.shape)
            assert np.array_equal(act, V)
        W = apply_gram(V, [symmetric_gram_numerator(k), symmetric_gram_numerator(l)]).T
        out.append(dict(tag=tag, k=k, l=l, W=W))
    return out


def ico_integer_states():
    u = np.zeros((2,) * 7, dtype=np.int64)
    v = np.zeros_like(u)
    pent = {(1, 3), (2, 3), (2, 4), (4, 5), (1, 5)}
    for pair in itertools.combinations(range(6), 2):
        sign = -1 if pair in pent else 1
        x = [0] * 7
        for k in pair:
            x[1 + k] = 1
        u[tuple(x)] = sign
        x = [1] * 7
        for k in pair:
            x[1 + k] = 0
        u[tuple(x)] = -sign
    v[(0, 1, 1, 1, 1, 1, 1)] = v[(1, 0, 0, 0, 0, 0, 0)] = 1
    assert np.count_nonzero(u) == 30 and np.count_nonzero(v) == 2
    assert np.array_equal(u, u.transpose(CYCLE)) and np.array_equal(
        v, v.transpose(CYCLE)
    )
    return u.ravel(), v.ravel()


def local_map_m2():
    J = sp.zeros(32)
    for x in range(32):
        for c in range(5):
            mask = 1 << (4 - c)
            if not (x & mask):
                J[x | mask, x] += 1
    V = s5.V32[2]
    S = sp.Matrix.hstack(
        *[(V if k == 0 else J * V)[:, i] for i in range(5) for k in range(2)]
    )
    L = (S.T * S).inv() * S.T
    den = int(sp.ilcm(*[int(x.q) for x in L]))
    M = np.array([[int(den * x) for x in row] for row in L.tolist()], dtype=np.int64)
    assert sp.Matrix(M) * S == den * sp.eye(10)
    assert M.shape == (10, 32) and int(np.abs(M).max()) == 3
    return M.reshape(5, 2, 32)


def reps(l):
    seen, rr = set(), []
    for s in itertools.combinations(range(7), l):
        if s in seen:
            continue
        rr.append(s)
        v = s
        for _ in range(5):
            seen.add(v)
            v = tuple(sorted(CYCLE[i] for i in v))
    assert len(seen) == math.comb(7, l)
    return rr


def build_sym_transitions(max_deg=7):
    trans = {}
    for d in range(max_deg):
        m_cur = monomials(d)
        m_nxt = monomials(d + 1)
        idx_nxt = {m: i for i, m in enumerate(m_nxt)}
        table = np.zeros((len(m_cur), 5), dtype=np.int64)
        for i, m in enumerate(m_cur):
            for c in range(5):
                mc = list(m)
                mc[c] += 1
                table[i, c] = idx_nxt[tuple(mc)]
        trans[d] = (len(m_nxt), table)
    return trans


def sym_chain(vecs, sites, trans):
    N = vecs.shape[0]
    cur = np.ones((N, 1), dtype=np.int64)
    for d, s in enumerate(sites):
        n_nxt, table = trans[d]
        nxt = np.zeros((N, n_nxt), dtype=np.int64)
        sv = vecs[:, s, :]
        for c in range(5):
            nxt[:, table[:, c]] += cur * sv[:, c : c + 1]
        cur = nxt
    return cur


def contract_support_multisets(u, v, models_out):
    M2 = local_map_m2()
    supp = [x for x in range(128) if u[x] != 0 or v[x] != 0]
    assert len(supp) == 32
    bits_supp = np.array(
        [[(x >> (6 - s)) & 1 for s in range(7)] for x in supp], dtype=np.int64
    )
    u_supp = np.array([int(u[x]) for x in supp], dtype=np.int64)
    v_supp = np.array([int(v[x]) for x in supp], dtype=np.int64)

    supp_set = set(supp)
    charge = lambda x: (x & 63).bit_count() + 2 * (x >> 6)
    assert supp_set == {x for x in range(128) if charge(x) % 4 == 2}
    # This is a Z/4 charge condition, not an affine binary code.
    assert all(x in supp_set for x in (3, 5, 6)) and (3 ^ 5 ^ 6) not in supp_set
    valid_by_b = {b: [] for b in supp}
    total_valid = 0
    for comb in itertools.combinations_with_replacement(range(32), 5):
        w = bits_supp[list(comb)].sum(axis=0)
        if np.all((w == 2) | (w == 3)):
            b_bits = w - 2
            b_int = int(sum(int(b_bits[s]) << (6 - s) for s in range(7)))
            assert (int(w[1:].sum()) + 2 * int(w[0])) % 4 == 2
            assert charge(b_int) % 4 == 2
            assert b_int in supp_set
            valid_by_b[b_int].append(comb)
            total_valid += 1
    assert total_valid == 14432
    assert len(valid_by_b[3]) == 466 and len(valid_by_b[63]) == 226
    assert all(len(valid_by_b[b]) == (466 if u[b] else 226) for b in supp)

    trans = build_sym_transitions(7)
    C_by_tag = {
        "7": np.zeros((6, 4, 128), dtype=object),
        "61": np.zeros((6, 6, 384), dtype=object),
        "52": np.zeros((6, 9, 640), dtype=object),
    }

    for b_int, combs in valid_by_b.items():
        b_bits = [(b_int >> (6 - s)) & 1 for s in range(7)]
        N_b = len(combs)
        combs_arr = np.array(combs, dtype=np.int64)
        orbits = np.zeros(N_b, dtype=np.int64)
        for idx_c, comb in enumerate(combs):
            cnts = {}
            for x in comb:
                cnts[x] = cnts.get(x, 0) + 1
            orbits[idx_c] = 120 // math.prod(math.factorial(c) for c in cnts.values())
        u_vals = u_supp[combs_arr]
        v_vals = v_supp[combs_arr]
        poly = np.zeros((N_b, 6), dtype=np.int64)
        for mask in range(32):
            d = mask.bit_count()
            prod = np.ones(N_b, dtype=np.int64)
            for c in range(5):
                prod *= v_vals[:, c] if ((mask >> c) & 1) else u_vals[:, c]
            poly[:, d] += prod
        poly *= orbits[:, None]

        site_vecs = np.zeros((N_b, 7, 5), dtype=np.int64)
        for s in range(7):
            r_s = (
                (bits_supp[combs_arr[:, 0], s] << 4)
                | (bits_supp[combs_arr[:, 1], s] << 3)
                | (bits_supp[combs_arr[:, 2], s] << 2)
                | (bits_supp[combs_arr[:, 3], s] << 1)
                | bits_supp[combs_arr[:, 4], s]
            )
            site_vecs[:, s, :] = M2[:, b_bits[s], r_s].T

        for M in models_out:
            tag, k, l, W = M["tag"], M["k"], M["l"], M["W"]
            for r_i, sub in enumerate(reps(l)):
                sub_set = set(sub)
                k_sites = [s for s in range(7) if s not in sub_set]
                l_sites = list(sub)
                sym_k = sym_chain(site_vecs, k_sites, trans)
                if l == 0:
                    sym_kl = sym_k
                else:
                    sym_l = sym_chain(site_vecs, l_sites, trans)
                    sym_kl = (sym_k[:, :, None] * sym_l[:, None, :]).reshape(N_b, -1)
                coeff_mat = (poly.T @ sym_kl).astype(object)
                feat = coeff_mat @ W.T
                col = r_i * 128 + b_int
                C_by_tag[tag][:, :, col] = feat
    return C_by_tag


def main():
    start = time.time()
    parity_inputs = sorted(DATA.glob("five_copy_ico_g34_pair*.npz"))
    assert len(parity_inputs) == 4
    for filename in parity_inputs:
        pair = np.load(filename)
        assert not np.any(
            np.any(pair["A"] != 0, axis=0) & np.any(pair["B"] != 0, axis=0)
        )
        aa = flint.fmpz_mat(pair["A"].tolist())
        bb = flint.fmpz_mat(pair["B"].tolist())
        assert aa * bb.transpose() == flint.fmpz_mat(aa.nrows(), bb.nrows())
    print(
        "TT-1 input check: even and odd parts have disjoint column supports (so A B^T = 0) "
        "in all four stored IK5 arrays",
        flush=True,
    )
    u, v = ico_integer_states()
    # D multiplies |s,a> by omega^e, omega = exp(i pi/4), e = (6 if s == 0 else 4) + |a| mod 8.
    expo = [((6 if x >> 6 == 0 else 4) + (x & 63).bit_count()) % 8 for x in range(128)]
    assert all(expo[x] == 0 for x in range(128) if u[x]) and all(
        expo[x] == 4 for x in range(128) if v[x]
    )
    print(
        "TT-1 symmetry: D u = u and D v = -v for the diagonal local unitary D",
        flush=True,
    )
    models_out = build_unflipped_models()
    C_by_tag = contract_support_multisets(u, v, models_out)

    faces = json.loads(
        (DATA / "five_copy_ico_unflipped_oneflip_physical_faces.json").read_text()
    )["faces"]
    for M in models_out:
        tag = M["tag"]
        C = C_by_tag[tag]
        even_cols = np.any(C[0::2] != 0, axis=(0, 1))
        odd_cols = np.any(C[1::2] != 0, axis=(0, 1))
        assert not np.any(even_cols & odd_cols)
        A = 225 * C[0] + 555 * C[2] + 1369 * C[4]
        B = 225 * C[1] + 555 * C[3] + 1369 * C[5]
        assert np.all((A @ B.T) == 0)
        rec = [f for f in faces if f["flip"] == 0 and f["tag"] == tag]
        assert len(rec) == 1
        K = np.array(rec[0]["physical_kernel_rows"], dtype=object)
        assert np.all((K @ A) == 0) and np.all(
            (K @ B) == 0
        ), "kernel rows must annihilate X(t*)"
        print(
            "TT-1 face j=0 shape",
            tag,
            "kernel rows",
            K.shape[0],
            "feature columns",
            A.shape[1],
            ": even/odd columns disjoint; both kernel components vanish exactly",
            flush=True,
        )
    print(
        "TT-1 PASS: exact column parity and physical kernel annihilation; sec",
        round(time.time() - start),
        flush=True,
    )


if __name__ == "__main__":
    main()
