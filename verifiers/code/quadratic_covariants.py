"""Exact six-copy covariants for the quadratic purity-variance certificate.

Five-copy components are shared with the joint-purity certificate. Recoupling
adds one spin 1/2; integer contractions carry explicit overflow bounds.
"""
from itertools import product
from math import comb, lcm
import numpy as np
from flint import fmpz_mat
from spin_covariants import Covariants, ORDERS, contract


def integer_gram(X, Y, weights):
    X, Y = X.reshape(-1, X.shape[-1]), Y.reshape(-1, Y.shape[-1])
    weights = np.tile(weights, len(ORDERS))
    bound = max(int(np.abs(X).max()), int(np.abs(Y).max()))
    chunk = min(len(X), (2**61)//max(1, 2*bound**2*int(max(weights))))
    assert chunk > 0
    G = np.zeros((X.shape[1], X.shape[1]), dtype=object)
    for k in range(0, len(X), chunk):
        a, b, w = X[k:k+chunk], Y[k:k+chunk], weights[k:k+chunk]
        assert 2*bound**2*int(sum(w)) < 2**62
        G += a.T@(w[:, None]*a)+b.T@(w[:, None]*b)
    return fmpz_mat(G.tolist())


class QuadraticCovariants:
    def __init__(self, data):
        self.five = Covariants(data['five_copy'])
        self.parents, self.others = [], []
        for i, (_, p, _, _) in enumerate(self.five.blocks):
            (self.parents if p in ((1, 1, 1, 1), (1, 1, 1, 3))
             else self.others).append(i)
        self.R = {}
        for j in (1, 3):
            R = np.zeros((2*j+2, 2*j+2), dtype=np.int64)
            for k in range(j):
                R[k, 2*k+1], R[k, 2*k+2] = j-k, -k-1
            for k in range(j+2):
                if k <= j:
                    R[j+k, 2*k] = 1
                if k >= 1:
                    R[j+k, 2*k-1] = 1
            self.R[j] = R
        self.transforms = {}
        for block in data['blocks']:
            B = np.zeros(block['shape'], dtype=np.int64)
            for i, j, v in block['basis']:
                B[i, j] = v
            self.transforms[block['raw']] = fmpz_mat(B.tolist())

    def components(self, z0, z1, omega=False):
        tensors = {a: self.five.tensor(z0, z1, a, omega) for a in (3, 4, 5)}
        old = []
        for a, p, K, _ in self.five.blocks:
            old.append(tuple(np.array([self.five.components(w, p, o, K)
                                       for o in ORDERS]) for w in tensors[a]))
        out = [old[i] for i in self.others]
        weights = [self.five.blocks[i][3] for i in self.others]
        groups = {}
        signs = np.array([(-1)**i.bit_count() for i in range(16)])
        flip = ((z0-z1 if omega else z0)[np.arange(16)^15]*signs,
                -z1[np.arange(16)^15]*signs)
        extra_states = [tuple(np.array([v.reshape([2]*4).transpose(o).reshape(16)
                                       for o in ORDERS]) for v in pair)
                        for pair in ((z0, z1), flip)]
        for i in self.parents:
            a, p, _, _ = self.five.blocks[i]
            X, Y = old[i]
            for (u, v), newa in zip(extra_states, (a+1, a)):
                bound = max(int(abs(X).max()), int(abs(Y).max()))
                assert 3*bound*max(int(abs(u).max()), int(abs(v).max())) < 2**62
                bd = Y[:, :, None, :]*v[:, None, :, None]
                pair = (X[:, :, None, :]*u[:, None, :, None]-bd,
                        X[:, :, None, :]*v[:, None, :, None]
                        +Y[:, :, None, :]*u[:, None, :, None]-(bd if omega else 0))
                recoupled = []
                for w in pair:
                    w = w.reshape([24]+[j+1 for j in p]+[2]*4+[X.shape[-1]])
                    w = w.transpose(0, 1, 5, 2, 6, 3, 7, 4, 8, 9)
                    w = w.reshape([24]+[2*j+2 for j in p]+[X.shape[-1]])
                    for axis, j in enumerate(p, 1):
                        w = contract(self.R[j], w, axis)
                    recoupled.append(w)
                seen = set()
                for choices in product((0, 1), repeat=4):
                    q = tuple(j+(-1 if c == 0 else 1) for j, c in zip(p, choices))
                    order = tuple(sorted(range(4), key=lambda k: q[k]))
                    spins = tuple(q[k] for k in order)
                    if spins in seen:
                        continue
                    seen.add(spins)
                    ix = [np.arange(j) if c == 0 else np.arange(j, 2*j+2)
                          for j, c in zip(p, choices)]
                    cs = []
                    for w in recoupled:
                        c = w[:, ix[0]][:, :, ix[1]][:, :, :, ix[2]][:, :, :, :, ix[3]]
                        cs.append(c.transpose((0,)+tuple(k+1 for k in order)+(5,))
                                  .reshape(24, -1, X.shape[-1]))
                    groups.setdefault((newa, spins), []).append(cs)
        for key in sorted(groups):
            out.append(tuple(np.concatenate([c[k] for c in groups[key]], axis=-1)
                             for k in (0, 1)))
            wt = np.array([1], dtype=np.int64)
            for j in key[1]:
                d = lcm(*(comb(j, k) for k in range(j+1)))
                wt = np.kron(wt, [d//comb(j, k) for k in range(j+1)])
            weights.append(wt)
        return out, weights

    def features(self, z0, z1):
        cs, weights = self.components(z0, z1)
        n = sum(int(x)**2+int(y)**2 for x, y in zip(z0, z1))
        raw, final = [], []
        for i, ((X, Y), w) in enumerate(zip(cs, weights)):
            G = integer_gram(X, Y, w)
            if i < len(self.others):
                G *= n
            raw.extend(int(G[a, b])*(1 if a == b else 2)
                       for a in range(G.nrows()) for b in range(a, G.nrows()))
            if i in self.transforms:
                B = self.transforms[i]
                H = B.transpose()*G*B
                final.extend(int(H[a, b])*(1 if a == b else 2)
                             for a in range(H.nrows()) for b in range(a, H.nrows()))
        return raw, final

    def check_zeros(self):
        hs0 = np.zeros(16, dtype=np.int64)
        hs1 = hs0.copy()
        hs0[[3, 12]], hs1[[5, 10]] = 1, 1
        hs0[[6, 9]], hs1[[6, 9]] = -1, -1
        cl = hs0*0
        cl[[0, 3, 12, 15]] = [1, 1, 1, -1]
        for pair in ((hs0, hs1), (cl, cl*0)):
            for a in (3, 4, 5):
                tensors = self.five.tensor(*pair, a, omega=True)
                for aa, p, K, _ in self.five.blocks:
                    if a == aa:
                        assert all(not self.five.components(w, p, o, K).any()
                                   for w in tensors for o in ORDERS)
