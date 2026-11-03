"""Floating proposals for the entropy Hessian in Schmidt amplitudes.

Every use as a certificate requires independent SVD residual, coefficient,
arithmetic-error and positivity checks, as in certify_thirteen_entropy_local.py.
"""

import numpy as np, math
from itertools import combinations


class EntropyAmplitudeHessian:
    def __init__(self, psi, n=13, k=6):
        self.n = n
        self.k = k
        self.psi = psi / np.linalg.norm(psi)
        self.cache = []
        Es = []
        self.least = 1.0
        h = np.zeros_like(self.psi.reshape([2] * n))
        ln2 = np.log(2.0)
        for S in combinations(range(n), k):
            q = [n - 1 - i for i in S] + [n - 1 - i for i in range(n) if i not in S]
            iq = np.argsort(q)
            M = self.psi.reshape([2] * n).transpose(q).reshape(1 << k, -1)
            U, s, Vh = np.linalg.svd(M, full_matrices=False)
            p = s * s
            logs = np.log2(p)
            Es.append(-p @ logs)
            self.least = min(self.least, p[-1])
            h += (-(U * logs) @ (s[:, None] * Vh)).reshape([2] * n).transpose(iq)
            t = (s[:, None] - s[None, :]) / (s[:, None] + s[None, :])
            at = np.arctanh(t)
            ratio = np.divide(at, t, out=np.ones_like(t), where=abs(t) > 0)
            ll = (np.log(s)[:, None] + np.log(s)[None, :]) + 1
            a = -2 * (ll + 2 * ratio) / ln2
            b = -2 * (ll + 2 * t * at) / ln2
            kdiag = -2 * (logs + 1 / ln2)
            self.cache.append((q, iq, U, s, Vh, kdiag, a, b))
        self.E = math.fsum(Es) / len(Es)
        self.gradient = 2 * (h.ravel() / len(Es) - self.E * self.psi)

    def action(self, eta):
        T = eta.reshape([2] * self.n)
        out = np.zeros_like(T)
        for q, iq, U, s, Vh, k, a, b in self.cache:
            B = T.transpose(q).reshape(1 << self.k, -1)
            Bt = U.conj().T @ B
            C = Bt @ Vh.conj().T
            Ct = C.conj().T
            D = a * ((C + Ct) / 2) + b * ((C - Ct) / 2) - k[:, None] * C
            out += (U @ (k[:, None] * Bt + D @ Vh)).reshape([2] * self.n).transpose(iq)
        return out.ravel() / len(self.cache) - 2 * (self.E - 1 / np.log(2.0)) * eta
