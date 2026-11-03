"""PK5-3: exact unflipped/one-flip all-V ICO kernels, without saved maps."""
import argparse, json, time, math
from pathlib import Path
import numpy as np, sympy as sp, flint
from five_copy_unflipped_oneflip_faces import models, physical_tensor
from five_copy_physical_faces import apply_gram, symmetric_gram_numerator, integer_matrix, short_kernel
from certify_ico_fivecopy_g34_51 import ico_integer_states
from certify_ico_fivecopy_allv_faces import reps, row_sym
P = 2147483629
DATA = Path(__file__).resolve().parent.parent / 'data'
ARTIFACT = DATA / 'five_copy_ico_unflipped_oneflip_physical_faces.json'

def fm(X):
    return integer_matrix(X)

def lists(X):
    return [[int(X[i, j]) for j in range(X.ncols())] for i in range(X.nrows())]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    expected = None if args.write else json.loads(ARTIFACT.read_text())['faces']
    start = time.time()
    ms = []
    for flip, tag, k, l, U in models():
        W = integer_matrix(apply_gram(U, [symmetric_gram_numerator(k), symmetric_gram_numerator(l)])).transpose()
        ms.append(dict(flip=flip, tag=tag, k=k, l=l, U=U, W=W, X=[]))
    u, v = ico_integer_states()
    VV = sp.Matrix([[t ** i for i in range(6)] for t in range(6)]).inv() * 120
    assert all(v.q == 1 for v in VV)
    records = []
    for flip in [0, 1]:
        for t in range(6):
            T = physical_tensor(u + t * v, flip)
            for M in ms:
                if M['flip'] != flip:
                    continue
                k, l, U, W = (M['k'], M['l'], M['U'], M['W'])
                Ys = [row_sym(T, k, l, s) for s in reps(l)]
                Xs = [W * fm(Y) for Y in Ys]
                nc = sum((Y.ncols() for Y in Xs))
                M['X'].append(flint.fmpz_mat(U.shape[1], nc, [Y[i, j] for i in range(U.shape[1]) for Y in Xs for j in range(Y.ncols())]))
            del T
            print('physical flip', flip, 'input', t, 'sec', time.time() - start, flush=True)
        for M in ms:
            if M['flip'] != flip:
                continue
            Xs = M['X']
            C = []
            for i in range(6):
                X = sum((int(VV[i, t]) * Xs[t] for t in range(6)), flint.fmpz_mat(Xs[0].nrows(), Xs[0].ncols()))
                assert all((int(x) % 120 == 0 for x in X.entries()))
                C.append(flint.fmpz_mat(X.nrows(), X.ncols(), [int(x) // 120 for x in X.entries()]))
            for t in range(6):
                assert sum((t**i*C[i] for i in range(6)), flint.fmpz_mat(Xs[0].nrows(), Xs[0].ncols())) == Xs[t]
            A = 225 * C[0] + 555 * C[2] + 1369 * C[4]
            B = 225 * C[1] + 555 * C[3] + 1369 * C[5]
            common = flint.fmpz_mat(A.nrows(), 2 * A.ncols(), [Y[i, j] for i in range(A.nrows()) for Y in (A, B) for j in range(A.ncols())])
            rank = common.rank()
            root = int(sp.sqrt_mod(555, P))
            field = flint.nmod_mat(A.nrows(), A.ncols(), [(15 * int(A[i, j]) + root * int(B[i, j])) % P for i in range(A.nrows()) for j in range(A.ncols())], P)
            assert field.rank() == rank
            R, r = flint.nmod_mat(common.nrows(), common.ncols(), [int(x) % P for x in common.entries()], P).rref()
            J = [next((j for j in range(common.ncols()) if R[i, j])) for i in range(r)]
            Q = flint.fmpz_mat(common.nrows(), r, [common[i, j] for i in range(common.nrows()) for j in J])
            K = short_kernel(Q, rank)
            assert K * common == flint.fmpz_mat(K.nrows(), common.ncols())
            rec = dict(tag=M['tag'], flip=flip, rank=rank, size=M['U'].shape[1], physical_kernel_rows=lists(K), physical_columns=lists(Q))
            records.append(rec)
            if expected is not None:
                assert rec == expected[len(records) - 1]
            print('new physical face', flip, M['tag'], 'rank', rank, 'kernel', K.nrows(), 'sec', time.time() - start, flush=True)
    if args.write:
        ARTIFACT.write_text(json.dumps(dict(faces=records), separators=(',', ':')) + '\n')
    print('PK5-3 PASS: six exact physical kernel bases; no sharp dual identity asserted', flush=True)
if __name__ == '__main__':
    main()
