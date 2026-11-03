#!/usr/bin/env python3
"""ST-7, ST-8: odd-n purity--time-reversal identity (spinflip_tradeoff.tex, thm:odd-spinflip-tradeoff).

For a pure n-qubit state, n odd, let A_j be the Pauli weight distribution, pi_m the mean purity of the
m = (n-1)/2 qubit marginals, and O the mean over single qubits s of the T-odd purity
(tr rho^2 - R)/2 of the (n-1)-qubit complement, R = tr(rho Y^k conj(rho) Y^k).  With c_5 = 1, c_n = (n+1)/4 (n >= 7),
    pi_m + c_n O = 1/2 + sum_{j odd, j<n} w_j A_j,   w_j >= 0 rational, w_{n-2} = 0.
ST-7: the identity of linear forms in A (exact rational arithmetic, odd n = 5..41), using only
      A_0 = 1 and the complementary-purity relations a_s = a_{n-s}; the weights w_j are positive on odd j <= n-4
      and zero at j = n-2 (so c_n is least for these multipliers), and the elimination reproduces the closed forms of e_s and w_j stated in the theorem (proved there for all odd n).
ST-8: exact evaluation of both sides on Gaussian-integer states (n = 5, 7, 9) from reduced density matrices
      and direct Pauli expectations, which does not use the formulas for a_s and O in terms of A_j.
"""
import itertools
from fractions import Fraction as F
from math import comb
import numpy as np

def ch(n, k): return comb(n, k) if 0 <= k <= n else 0
def a_coeff(n, s, j): return F(ch(n - j, s - j), 2**s * comb(n, s))

def c_coeff(n): return F(1) if n == 5 else F(n + 1, 4)

def certificate(n):
    """rational e_0..e_m and w_j (j odd) with  T - A_0/2 = sum_s e_s (a_s - a_{n-s}) + sum w_j A_j."""
    m = (n - 1) // 2
    T = [a_coeff(n, m, j) for j in range(n + 1)]
    for j in range(1, n + 1, 2):
        T[j] += c_coeff(n) * F(n - j, n * 2**(n - 1))       # c_n * O, O = sum_odd (n-j)/(n 2^(n-1)) A_j
    T[0] -= F(1, 2)
    def rel(s, j):                                           # coefficient of A_j in a_s - a_{n-s}; s = 0: A_0 - 2^-n sum A
        return (1 if j == 0 else 0) - F(1, 2**n) if s == 0 else a_coeff(n, s, j) - a_coeff(n, n - s, j)
    M = [[rel(s, j) for s in range(m + 1)] + [T[j]] for j in range(0, n + 1, 2)]   # even-j equations fix e
    for c in range(m + 1):
        p = next(i for i in range(c, m + 1) if M[i][c] != 0)
        M[c], M[p] = M[p], M[c]
        M[c] = [x / M[c][c] for x in M[c]]
        for i in range(m + 1):
            if i != c and M[i][c] != 0:
                f = M[i][c]
                M[i] = [x - f * y for x, y in zip(M[i], M[c])]
    e = [M[i][m + 1] for i in range(m + 1)]
    return e, {j: T[j] - sum(e[s] * rel(s, j) for s in range(m + 1)) for j in range(1, n + 1, 2)}

def e_closed(n, s):
    m = (n - 1) // 2
    if s < 2: return F(0)
    return (-1)**(s - 1) * (F(comb(n, s), 2) - ch(m, s) - F(ch(m, s - 1), 2)) + (1 if s == m and m % 2 == 0 else 0)

def w_closed(n, j):
    m, k = (n - 1) // 2, (n - j) // 2
    N = 2**m * (ch(m, j - 1) + 3 * ch(m, j)) + (-1)**k * ch(m, k) + int(2 * c_coeff(n)) * ch(n - 1, j) - comb(n, j)
    return F(N, 2**n * comb(n, j))

def st7():
    for n in range(5, 42, 2):
        m = (n - 1) // 2
        e, w = certificate(n)
        assert all(e[s] == e_closed(n, s) for s in range(m + 1)), (n, e)
        assert all(w[j] == w_closed(n, j) for j in w), (n, w)
        for j in range(n + 1):   # every coefficient of the identity, even and odd j
            res = a_coeff(n, m, j) - (F(1, 2) if j == 0 else 0) - sum(
                e[s] * (((1 if j == 0 else 0) - F(1, 2**n)) if s == 0 else a_coeff(n, s, j) - a_coeff(n, n - s, j))
                for s in range(m + 1))
            if j % 2:
                res += c_coeff(n) * F(n - j, n * 2**(n - 1)) - w[j]
            assert res == 0, (n, j)
        assert all(v >= 0 for v in w.values()), (n, w)
        support = [j for j, v in w.items() if v]
        assert support == list(range(1, n - 3, 2)), (n, support)
        assert w[n - 2] == 0 and w[n] == 0, n   # any smaller O coefficient makes w_{n-2} negative
        if n <= 11:
            print('ST-7 PASS: n=%d w_j (j odd, j<n) = %s, w_n = %s' % (
                n, ', '.join('%d:%s' % (j, v) for j, v in w.items() if j < n), w[n]))
    print('ST-7 PASS: all odd n = 5..41: identity coefficientwise, e_s and w_j equal the closed forms, w_j >= 0, support = odd j <= n-4, w_{n-2} = w_n = 0')

# ---- ST-8: exact states ----
def wht(F_):                                                 # Walsh-Hadamard along the last axis, integers
    n = F_.shape[-1].bit_length() - 1
    for t in range(n):
        v = F_.reshape(F_.shape[:-1] + (-1, 2, 2**t))
        F_ = np.stack([v[..., 0, :] + v[..., 1, :], v[..., 0, :] - v[..., 1, :]], axis=-2).reshape(F_.shape)
    return F_
def weight_distribution(psr, psi_, n):
    N2 = int((psr**2 + psi_**2).sum())**2
    idx = np.arange(2**n)
    xr, xi = psr[idx[None, :] ^ idx[:, None]], psi_[idx[None, :] ^ idx[:, None]]   # psi[x xor a]
    Fr, Fi = xr * psr[None, :] + xi * psi_[None, :], xr * psi_[None, :] - xi * psr[None, :]   # conj(psi[x^a]) psi[x]
    P2 = wht(Fr)**2 + wht(Fi)**2                                                    # |<X^a Z^b>|^2 N^2
    wt = np.array([bin(v).count('1') for v in range(2**n)])[idx[:, None] | idx[None, :]]
    A = [F(int(P2[wt == j].sum()), N2) for j in range(n + 1)]
    assert A[0] == 1 and sum(A) == 2**n
    return A
def reduced(psr, psi_, n, keep):                             # integer G with rho = G / N
    rest = [i for i in range(n) if i not in keep]
    Mr = np.transpose(psr.reshape([2] * n), list(keep) + rest).reshape(2**len(keep), -1)
    Mi = np.transpose(psi_.reshape([2] * n), list(keep) + rest).reshape(2**len(keep), -1)
    return Mr @ Mr.T + Mi @ Mi.T, Mi @ Mr.T - Mr @ Mi.T
def Jk(k):                                                   # Y = i J, J = [[0,-1],[1,0]]
    J = np.array([[0, -1], [1, 0]], dtype=np.int64)
    out = np.array([[1]], dtype=np.int64)
    for _ in range(k): out = np.kron(out, J)
    return out
def purity_and_R(Gr, Gi, k, J, N2):
    pur = F(int((Gr**2 + Gi**2).sum()), N2)
    Kr, Ki = J @ Gr @ J, -(J @ Gi @ J)                       # J conj(G) J
    re, im = int((Gr * Kr.T - Gi * Ki.T).sum()), int((Gr * Ki.T + Gi * Kr.T).sum())
    assert im == 0
    return pur, F((-1)**k * re, N2)                          # Y^k conj(rho) Y^k = (-1)^k J^k conj(rho) J^k
def sides(psr, psi_, n):
    N2 = int((psr**2 + psi_**2).sum())**2
    m, k = (n - 1) // 2, n - 1
    subs = list(itertools.combinations(range(n), m))
    pi = sum(F(int(sum((g**2).sum() for g in reduced(psr, psi_, n, c))), N2) for c in subs) / len(subs)
    J = Jk(k)
    O = F(0)
    for s in range(n):
        Gr, Gi = reduced(psr, psi_, n, [i for i in range(n) if i != s])
        pur, R = purity_and_R(Gr, Gi, k, J, N2)
        O += (pur - R) / 2 / n
    A = weight_distribution(psr, psi_, n)
    _, w = certificate(n)
    return pi + c_coeff(n) * O, F(1, 2) + sum(w[j] * A[j] for j in w), A

def states(n):
    rng = np.random.default_rng(20261027 + n)
    for _ in range(3):
        yield 'random', rng.integers(-3, 4, 2**n).astype(np.int64), rng.integers(-3, 4, 2**n).astype(np.int64)
    z = np.zeros(2**n, dtype=np.int64)
    g = z.copy(); g[0] = g[-1] = 1
    yield 'GHZ', g, z
    ring = np.array([(-1)**sum(((x >> i) & 1) * ((x >> ((i + 1) % n)) & 1) for i in range(n)) for x in range(2**n)], dtype=np.int64)
    yield 'ring graph', ring, z
    w = np.zeros(2**n, dtype=np.int64)
    for i in range(n): w[1 << i] = 1
    yield 'W', w, z

def st8():
    for n in (5, 7, 9):
        out = []
        for name, pr, pi_ in states(n):
            lhs, rhs, A = sides(pr, pi_, n)
            if name == 'GHZ':
                assert all(A[j] == 0 for j in range(1, n, 2)) and lhs == F(1, 2), (n, A)   # equality case of the bound
            assert lhs == rhs, (n, name, lhs, rhs)
            out.append('%s %.6f' % (name, float(lhs)))
        print('ST-8 PASS: n=%d both sides equal exactly (rational): %s' % (n, '; '.join(out)))

if __name__ == '__main__':
    st7()
    st8()
