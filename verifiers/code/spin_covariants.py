"""Exact five-copy SU(2)^4 covariants for the joint-purity certificate.

Integer tensors use two coefficient arrays, in Q(i) or Q(omega), omega^2=-1-omega.
Every int64 contraction is preceded by an integer overflow bound.
"""
from itertools import permutations
from math import comb, lcm, prod
import numpy as np

ORDERS = tuple(permutations(range(4)))


def contract(M, w, axis):
    bound = int(np.abs(M).sum(axis=1).max()) * int(np.abs(w).max())
    assert bound < 2**62
    return np.moveaxis(np.tensordot(M, w, axes=(1, axis)), 0, axis)


class Covariants:
    def __init__(self, data):
        raising = np.zeros((32, 32), dtype=np.int64)
        for b in range(32):
            for k in range(5):
                if b >> k & 1:
                    raising[b ^ (1 << k), b] = 1
        rows, self.sizes, self.offsets = [], {}, {}
        for j in (1, 3, 5):
            H = np.array(data['highest_weights'][str(j)], dtype=np.int64)
            assert not (raising @ H.T).any()
            assert all(not H[:, b].any() for b in range(32)
                       if b.bit_count() != (5-j)//2)
            self.sizes[j] = (j+1, len(H))
            self.offsets[j] = sum(len(M) for M in rows)
            rows.append(H)
            for k in range(1, j+1):
                H = H @ raising
                assert not (H % k).any()
                H = H // k
                rows.append(H)
        self.C = np.concatenate(rows)
        self.blocks = []
        for block in data['blocks']:
            p = tuple(block['spins'])
            dimension = prod(self.sizes[j][1] for j in p)
            K = np.zeros((dimension, len(block['columns'])), dtype=np.int64)
            for k, column in enumerate(block['columns']):
                for i, value in column:
                    K[i, k] = value
            weights = np.array([1], dtype=np.int64)
            for j in p:
                d = lcm(*(comb(j, k) for k in range(j+1)))
                weights = np.kron(weights, [d//comb(j, k) for k in range(j+1)])
            self.blocks.append((block['a'], p, K, weights))

    def tensor(self, z0, z1, a, omega=False):
        z0, z1 = np.array(z0, dtype=np.int64), np.array(z1, dtype=np.int64)
        signs = np.array([(-1)**b.bit_count() for b in range(16)])
        flip0 = (z0-z1 if omega else z0)[np.arange(16) ^ 15]*signs
        flip1 = -z1[np.arange(16) ^ 15]*signs
        wa, wb = np.array([1], dtype=np.int64), np.array([0], dtype=np.int64)
        for k in range(5):
            u, v = (z0, z1) if k < a else (flip0, flip1)
            assert max(int(np.abs(wa).max()), int(np.abs(wb).max())) * max(
                int(np.abs(u).max()), int(np.abs(v).max())) * 3 < 2**62
            ac, bd = np.kron(wa, u), np.kron(wb, v)
            wa, wb = ac-bd, np.kron(wa, v)+np.kron(wb, u)-(bd if omega else 0)
        out = []
        for w in (wa, wb):
            w = w.reshape([2]*20).transpose(
                [4*c+s for s in range(4) for c in range(5)]).reshape([32]*4)
            for axis in range(4):
                w = contract(self.C, w, axis)
            out.append(w)
        return out

    def components(self, w, p, order, K):
        spins = [0]*4
        for k, i in enumerate(order):
            spins[i] = p[k]
        indices = [np.arange(self.offsets[j], self.offsets[j]+prod(self.sizes[j]))
                   for j in spins]
        c = w[np.ix_(*indices)].transpose(order)
        shape = sum(([self.sizes[j][0], self.sizes[j][1]] for j in p), [])
        c = c.reshape(shape).transpose(0, 2, 4, 6, 1, 3, 5, 7).reshape(-1, len(K))
        assert int(np.abs(c).max()) * int(np.abs(K).sum(axis=0).max()) < 2**62
        return c @ K

    def features(self, z0, z1):
        tensors = {a: self.tensor(z0, z1, a) for a in (3, 4, 5)}
        features = []
        for a, p, K, weights in self.blocks:
            G = np.zeros((K.shape[1], K.shape[1]), dtype=np.int64)
            for order in ORDERS:
                X, Y = [self.components(w, p, order, K) for w in tensors[a]]
                bound = max(int(np.abs(X).max()), int(np.abs(Y).max()))
                assert 48*bound**2*int(sum(weights)) < 2**62
                G += X.T@(weights[:, None]*X)+Y.T@(weights[:, None]*Y)
            ix = np.triu_indices(len(G))
            features.extend(int(v)*(1 if i == j else 2)
                            for i, j, v in zip(*ix, G[ix]))
        return features

    def check_zeros(self):
        hs0, hs1 = np.zeros(16, dtype=np.int64), np.zeros(16, dtype=np.int64)
        hs0[[3, 12]], hs1[[5, 10]] = 1, 1
        hs0[[6, 9]], hs1[[6, 9]] = -1, -1
        cl = np.zeros(16, dtype=np.int64)
        cl[[0, 3, 12, 15]] = [1, 1, 1, -1]
        bell = np.abs(cl)
        for z0, z1 in ((hs0, hs1), (cl, cl*0), (bell, bell*0)):
            tensors = {a: self.tensor(z0, z1, a, omega=True) for a in (3, 4, 5)}
            for a, p, K, _ in self.blocks:
                for order in ORDERS:
                    assert all(not self.components(w, p, order, K).any()
                               for w in tensors[a])
