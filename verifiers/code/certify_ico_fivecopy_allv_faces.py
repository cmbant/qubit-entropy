"""PK5-2: exact physical ICO kernels for the all-V [6,1] and [5,2] faces.

Contracts all spin columns at representatives of every distinguished-site
or pair orbit. Six integer amplitude ratios determine the degree-five
feature polynomial. The exact common A/B rank and a nonzero minor over
F_p(sqrt(555)) establish matching upper and lower physical ranks.
No full gamma maps, fitted matrices, or numerical rank thresholds enter.
--write creates the candidate artifact before the default full comparison.
"""
import math, itertools, time, json
import numpy as np, sympy as sp, flint
from pathlib import Path
import argparse
from five_copy_allv_young_models import invariant_basis
from five_copy_physical_faces import apply_gram, symmetric_gram_numerator, short_kernel
from certify_ico_fivecopy_g34_51 import ico_integer_states, spinflip, physical_sector, local_maps, CYCLE
from five_copy_symmetric_models import symmetric_lookup
from five_copy_young51_model import monomials
P = 2147483629

def reps(l):
    seen = set()
    rr = []
    for s in itertools.combinations(range(7), l):
        if s in seen:
            continue
        rr.append(s)
        v = s
        for _ in range(5):
            seen.add(v)
            v = tuple(sorted((CYCLE[i] for i in v)))
    assert len(seen) == math.comb(7, l)
    return rr

def row_sym(T, k, l, s):
    order = [i for i in range(7) if i not in s] + list(s)
    X = T.transpose(order + list(range(7, 14))).reshape(5 ** k, 5 ** l, 128)
    o, starts = symmetric_lookup(k)
    assert math.factorial(k) * math.factorial(l) * int(abs(X).max()) < 2 ** 63
    X = np.add.reduceat(X[o], starts, axis=0)
    if l > 1:
        o, starts = symmetric_lookup(l)
        X = np.add.reduceat(X[:, o], starts, axis=1)
    return X.reshape(len(monomials(k)) * len(monomials(l)), 128)

def fm(X):
    return flint.fmpz_mat(X.shape[0], X.shape[1], [int(x) for x in X.ravel()])

def tolist(X):
    return [[int(X[i, j]) for j in range(X.ncols())] for i in range(X.nrows())]
DATA = Path(__file__).resolve().parent.parent / 'data'
ARTIFACT = DATA / 'five_copy_ico_allv_physical_faces.json'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    expected = None if args.write else json.loads(ARTIFACT.read_text())['faces']
    t0 = time.time()
    ms = []
    for k, l in [(6, 1), (5, 2)]:
        U = invariant_basis(k, l)
        fs = [symmetric_gram_numerator(k), symmetric_gram_numerator(l)]
        HU = apply_gram(U, fs)
        W = fm(HU).transpose()
        ms.append((k, l, U, W, []))
        print('HU', k, l, 'maxbits', max((abs(int(v)).bit_length() for v in HU.ravel())), 'reps', reps(l), 'sec', time.time() - t0, flush=True)
    u, v = ico_integer_states()
    maps = local_maps()
    for t in range(6):
        # The shared contraction bound overestimates the all-V local maps.
        assert 32**5 * 5**5 * 3**7 < 2**53
        z = u + t * v
        T = physical_sector(z, spinflip(z), (2,) * 7, maps)
        T = T.reshape(*[5, 2] * 7).transpose([2 * i for i in range(7)] + [2 * i + 1 for i in range(7)])
        for k, l, U, W, Xs in ms:
            XX = [W * fm(row_sym(T, k, l, s)) for s in reps(l)]
            Xs.append(flint.fmpz_mat(U.shape[1], 128 * len(XX), [XX[j // 128][i, j % 128] for i in range(U.shape[1]) for j in range(128 * len(XX))]))
        print('physical input', t, 'sec', time.time() - t0, flush=True)
        del T
    V = sp.Matrix([[t ** i for i in range(6)] for t in range(6)])
    N = V.inv() * 120
    assert all((v.q == 1 for v in N))
    records = []
    for index, (k, l, U, W, Xs) in enumerate(ms):
        C = []
        for i in range(6):
            X = sum((int(N[i, t]) * Xs[t] for t in range(6)), flint.fmpz_mat(Xs[0].nrows(), Xs[0].ncols()))
            assert all((int(x) % 120 == 0 for x in X.entries()))
            C.append(flint.fmpz_mat(X.nrows(), X.ncols(), [int(x) // 120 for x in X.entries()]))
        for t in range(6):
            assert sum((t ** i * C[i] for i in range(6)), flint.fmpz_mat(Xs[0].nrows(), Xs[0].ncols())) == Xs[t]
        A = 225 * C[0] + 555 * C[2] + 1369 * C[4]
        B = 225 * C[1] + 555 * C[3] + 1369 * C[5]
        common = flint.fmpz_mat(A.nrows(), 2 * A.ncols(), [X[i, j] for i in range(A.nrows()) for X in (A, B) for j in range(A.ncols())])
        rank = common.rank()
        print('EXACT COMMON RANK', k, l, rank, flush=True)
        assert rank == 3
        R, r = flint.nmod_mat(common.nrows(), common.ncols(), [int(v) % P for v in common.entries()], P).rref()
        J = [next((j for j in range(common.ncols()) if R[i, j])) for i in range(r)]
        Q = flint.fmpz_mat(common.nrows(), r, [common[i, j] for i in range(common.nrows()) for j in J])
        K = short_kernel(Q, rank)
        assert K * common == flint.fmpz_mat(K.nrows(), common.ncols())
        root = int(sp.sqrt_mod(555, P))
        F = flint.nmod_mat(A.nrows(), A.ncols(), [(15 * int(A[i, j]) + root * int(B[i, j])) % P for i in range(A.nrows()) for j in range(A.ncols())], P)
        assert F.rank() == rank
        rec = dict(tag=str(k) + str(l), size=U.shape[1], rank=rank, gram_denominator=math.factorial(k) * math.factorial(l), source_columns=J, physical_columns=tolist(Q), physical_kernel_rows=tolist(K))
        records.append(rec)
        if expected is not None:
            assert rec == expected[index]
        print('KERNEL', k, l, K.nrows(), K.ncols(), 'max', max((abs(int(x)) for x in K.entries())), 'sec', time.time() - t0, flush=True)
    if args.write:
        ARTIFACT.write_text(json.dumps(dict(faces=records), separators=(',', ':')) + '\n')
    print('PK5-2 PASS: two exact physical rank-three kernels; dual identity and PSD remain OPEN', flush=True)
if __name__ == '__main__':
    main()
