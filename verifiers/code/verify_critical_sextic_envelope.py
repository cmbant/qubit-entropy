#!/usr/bin/env python3
"""Exact checks for the critical envelope F >= tau - tau^{3/2} (sextic_identity.tex, prop:fqe-critical-envelope).

CE-1: on psi = G(v), with v and its conjugate as independent symbols and Theta the Bloch correlation matrix of the
      auxiliary two-qubit vector phi = sum_k v_k m_k, the polynomial identities
        F = N^3/2 + (7/4) N tau + (9/4) perm Theta,   P = N^2 + tau/2,
        det Theta = -N tau,   charpoly(Theta^T Theta) = (x - N^2)(x - tau)^2,
      where N = |v|^2, tau = |sum v_k^2|^2 and F = 17NP - 12T - 12N^3.
CE-2: the cut-probability derivation: over the four sign triples with product -1,
      sum (s.g)^2 = 4|g|^2 and sum (s.g)^3 = -24 g1 g2 g3; with P = (3+|Theta|^2)/4 and
      T = (3 + 3|Theta|^2 - 6e)/16, e = (perm + det)/2, one gets F = -3/2 + 2|Theta|^2 + (9/2)e.
CE-3: perm(k I + (1-k) u u^T) = k^3 + k^2(1-k) + 2k(1-k)^2 e2(y) + 6(1-k)^3 e3(y), y_a = u_a^2, |u| = 1,
      and the Maclaurin value k^3 + k^2(1-k) + (2/3)k(1-k)^2 + (2/9)(1-k)^3 = (2 + 3k^2 + 4k^3)/9,
      and 1/2 + (7/4)k^2 - (9/4)(2+3k^2+4k^3)/9 = k^2 - k^3.
CE-4: among the 48 signed permutation matrices O, exactly 12 have det O = -1 and O^{(x)3} omega = -omega
      (omega the normalised symmetric e1e2e3 tensor): the signed cyclic permutations with sign product -1.
      For each of them and each cube diagonal s, perm(O(kI + (1-k)ss^T)) = -(2+3k^2+4k^3)/9 identically in k.
CE-5: numerical sanity only (not used in the proof): random critical states satisfy F >= tau - tau^{3/2}, and a state
      built from an equality matrix of CE-4 attains it.
"""
# log: verify_critical_sextic_envelope.txt
import itertools
import numpy as np
import sympy as sp
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

res = []


def check(tag, ok, msg):
    res.append(bool(ok))
    print(('[PASS] ' if ok else '[FAIL] ') + tag + ' ' + msg)


def perm3(A):
    return sum(A[0, p[0]] * A[1, p[1]] * A[2, p[2]] for p in itertools.permutations(range(3)))


I, s2 = sp.I, sp.sqrt(2)
m = [sp.Matrix([1, 0, 0, 1]) / s2, I * sp.Matrix([0, 1, 1, 0]) / s2,
     sp.Matrix([0, 1, -1, 0]) / s2, I * sp.Matrix([1, 0, 0, -1]) / s2]
mc = [x.conjugate() for x in m]
SIG = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -I], [I, 0]]), sp.Matrix([[1, 0], [0, -1]])]

# ---- CE-1 -----------------------------------------------------------------------------------------
v = sp.symbols('v1:5')
w = sp.symbols('w1:5')                      # w_k stands for conj(v_k)
psi = sum((v[k] * sp.kronecker_product(m[k], m[k]) for k in range(4)), sp.zeros(16, 1))
psic = sum((w[k] * sp.kronecker_product(mc[k], mc[k]) for k in range(4)), sp.zeros(16, 1))


def reduced(j):
    """Unnormalised reduced matrix on qubits (1, j)."""
    rest = [q for q in range(4) if q not in (0, j)]
    R = sp.zeros(4, 4)

    def idx(x, c):
        b = [0] * 4
        b[0], b[j], b[rest[0]], b[rest[1]] = x >> 1, x & 1, c >> 1, c & 1
        return 8 * b[0] + 4 * b[1] + 2 * b[2] + b[3]
    for a, bb in itertools.product(range(4), repeat=2):
        R[a, bb] = sp.expand(sum(psi[idx(a, c)] * psic[idx(bb, c)] for c in range(4)))
    return R


N = sp.expand(sum(v[k] * w[k] for k in range(4)))
P = T3 = 0
for j in (1, 2, 3):
    R = reduced(j)
    R2 = R * R
    P += R2.trace()
    T3 += (R2 * R).trace()
F = sp.expand(17 * N * P - 12 * T3 - 12 * N**3)
phi = sum((v[k] * m[k] for k in range(4)), sp.zeros(4, 1))
phic = sum((w[k] * mc[k] for k in range(4)), sp.zeros(4, 1))
Th = sp.Matrix(3, 3, lambda a, b: sp.expand((phic.T * sp.kronecker_product(SIG[a], SIG[b]) * phi)[0]))
tau = sp.expand(sum(x**2 for x in v) * sum(x**2 for x in w))
pT = sp.expand(perm3(Th))
check('CE-1', sp.expand(F - (N**3 / 2 + sp.Rational(7, 4) * N * tau + sp.Rational(9, 4) * pT)) == 0
      and sp.expand(P - (N**2 + tau / 2)) == 0,
      f'F = N^3/2 + (7/4)N tau + (9/4) perm Theta and P = N^2 + tau/2 on G(v) ({len(F.args)} monomials in F)')
x = sp.Symbol('x')
check('CE-1', sp.expand(Th.det() + N * tau) == 0
      and sp.expand((Th.T * Th).charpoly(x).as_expr() - (x - N**2) * (x - tau)**2) == 0,
      'det Theta = -N tau and Theta^T Theta has eigenvalues N^2, tau, tau')

# ---- CE-2 -----------------------------------------------------------------------------------------
g = sp.symbols('g1:4')
signs = [s for s in itertools.product((1, -1), repeat=3) if s[0] * s[1] * s[2] == -1]
dot = [sum(si * gi for si, gi in zip(s, g)) for s in signs]
ok2 = (len(signs) == 4 and sp.expand(sum(d**2 for d in dot) - 4 * sum(gi**2 for gi in g)) == 0
       and sp.expand(sum(d**3 for d in dot) + 24 * g[0] * g[1] * g[2]) == 0)
A = sp.Matrix(3, 3, sp.symbols('a1:10'))
cyc = sum(sp.prod([A[(b + k) % 3, b] for b in range(3)]) for k in range(3))
ok2 = ok2 and sp.expand(2 * cyc - perm3(A) - A.det()) == 0
n2, e = sp.symbols('n2 e')
Fcut = 17 * (3 + n2) / 4 - 12 * (3 + 3 * n2 - 6 * e) / 16 - 12
ok2 = ok2 and sp.expand(Fcut - (-sp.Rational(3, 2) + 2 * n2 + sp.Rational(9, 2) * e)) == 0
check('CE-2', ok2, 'sign-triple sums, cyclic diagonals = (perm+det)/2, and F = -3/2 + 2|Theta|^2 + (9/2)e')

# ---- CE-3 -----------------------------------------------------------------------------------------
k = sp.Symbol('k')
u = sp.symbols('u1:4')
y = [ui**2 for ui in u]
e2 = y[0] * y[1] + y[0] * y[2] + y[1] * y[2]
e3 = y[0] * y[1] * y[2]
U = sp.Matrix(3, 1, u)
pD = sp.expand(perm3(k * sp.eye(3) + (1 - k) * U * U.T))
form = k**3 + k**2 * (1 - k) * sum(y) + 2 * k * (1 - k)**2 * e2 + 6 * (1 - k)**3 * e3
target = k**3 + k**2 * (1 - k) + sp.Rational(2, 3) * k * (1 - k)**2 + sp.Rational(2, 9) * (1 - k)**3
ok3 = (sp.expand(pD - form) == 0 and sp.expand(target - (2 + 3 * k**2 + 4 * k**3) / 9) == 0
       and sp.expand(sp.Rational(1, 2) + sp.Rational(7, 4) * k**2 - sp.Rational(9, 4) * target - (k**2 - k**3)) == 0)
check('CE-3', ok3, 'perm(kI+(1-k)uu^T) expansion (sum y = 1 used afterwards), Maclaurin value (2+3k^2+4k^3)/9, '
      'and 1/2 + (7/4)k^2 - (9/4)(2+3k^2+4k^3)/9 = k^2 - k^3')

# ---- CE-4 -----------------------------------------------------------------------------------------
def omega_image(O):
    """Coefficients of O^{(x)3} applied to the symmetric e1e2e3 tensor, as a dict index-triple -> value."""
    out = {}
    for p in itertools.permutations(range(3)):
        cols = [O[:, p[t]] for t in range(3)]
        for idx in itertools.product(range(3), repeat=3):
            val = cols[0][idx[0]] * cols[1][idx[1]] * cols[2][idx[2]]
            if val != 0:
                out[idx] = out.get(idx, 0) + val
    return {kk: vv for kk, vv in out.items() if vv != 0}


omega = omega_image(sp.eye(3))
good = []
for p in itertools.permutations(range(3)):
    for eps in itertools.product((1, -1), repeat=3):
        O = sp.Matrix(3, 3, lambda a, b: eps[a] if p[b] == a else 0)
        img = omega_image(O)
        if O.det() == -1 and img == {kk: -vv for kk, vv in omega.items()}:
            good.append((p, eps, O))
cyclic = {(0, 1, 2), (1, 2, 0), (2, 0, 1)}
ok4 = len(good) == 12 and all(p in cyclic and eps[0] * eps[1] * eps[2] == -1 for p, eps, _ in good)
for _, _, O in good:
    for sg in itertools.product((1, -1), repeat=3):
        S = sp.Matrix(sg) / sp.sqrt(3)
        Th_eq = O * (k * sp.eye(3) + (1 - k) * S * S.T)
        ok4 = ok4 and sp.expand(perm3(Th_eq) + (2 + 3 * k**2 + 4 * k**3) / 9) == 0
check('CE-4', ok4, f'{len(good)} signed permutations with det -1 and O^(x3) omega = -omega, all signed cyclic with '
      'sign product -1; perm Theta = -(2+3k^2+4k^3)/9 on all 12 x 8 equality matrices')

# ---- CE-5 (numerical sanity, not part of the proof) ------------------------------------------------
Fn = sp.lambdify([v, w], F, 'numpy')
taun = sp.lambdify([v, w], tau, 'numpy')
Thn = sp.lambdify([v, w], Th, 'numpy')
rng = np.random.default_rng(20261029)
worst = np.inf
for _ in range(3000):
    z = rng.normal(size=4) + 1j * rng.normal(size=4)
    z /= np.linalg.norm(z)
    t = taun(z, z.conj()).real
    worst = min(worst, Fn(z, z.conj()).real - t + t**1.5)
# equality state: phi with Bloch correlation O(kI+(1-k)ss^T), O = -I, s = (1,1,1)/sqrt3, k = 3/5
kk = 0.6
s_dir = np.ones(3) / np.sqrt(3)
target_T = -(kk * np.eye(3) + (1 - kk) * np.outer(s_dir, s_dir))
Mg = np.array(sp.matrix2numpy(sp.Matrix.hstack(*m).evalf()), dtype=complex)
Pauli = [np.array(sp.matrix2numpy(s.evalf()), dtype=complex) for s in SIG]
# closed form: phi = (U (x) V)(cos a |00> + sin a |11>) has Bloch correlation R_a diag(sin2a,-sin2a,1) R_b^T;
# choose the frame so that this equals target_T and pull back to v = M^dagger phi.
Uq, sv, Vq = np.linalg.svd(target_T)


def su2_of(Rm):
    """An SU(2) element whose adjoint action is the rotation Rm."""
    rv = Rotation.from_matrix(Rm).as_rotvec()
    return expm(-0.5j * sum(rv[c] * Pauli[c] for c in range(3)))


# target_T = R_a diag(kk,-kk,1) R_b^T with R_a, R_b in SO(3)
Ra = Uq[:, [1, 2, 0]] @ np.diag([1, -1, 1])
Rb = Vq.T[:, [1, 2, 0]]
if np.linalg.det(Ra) < 0:
    Ra, Rb = -Ra, -Rb
if np.linalg.det(Rb) < 0:
    Ra, Rb = Ra @ np.diag([-1, -1, 1]), Rb @ np.diag([1, 1, -1])
assert np.allclose(Ra @ np.diag([kk, -kk, 1]) @ Rb.T, target_T, atol=1e-12)
a_ang = 0.5 * np.arcsin(kk)
phi0 = np.array([np.cos(a_ang), 0, 0, np.sin(a_ang)], dtype=complex)
phin = np.kron(su2_of(Ra), su2_of(Rb)) @ phi0
vn = Mg.conj().T @ phin
Theq = np.array(Thn(vn, vn.conj()), dtype=complex).real
teq = taun(vn, vn.conj()).real
gap = Fn(vn, vn.conj()).real - teq + teq**1.5
check('CE-5', worst > -1e-12 and np.allclose(Theq, target_T, atol=1e-10) and abs(gap) < 1e-10,
      f'[numerical sanity] min of F - tau + tau^(3/2) over 3000 random critical states = {worst:.3e}; '
      f'equality state with Theta = -(kI+(1-k)ss^T), k = 3/5: tau = {teq:.6f}, gap = {gap:.1e}')

print('ALL CE CHECKS PASS' if all(res) else 'SOME CE CHECK FAILED')
raise SystemExit(0 if all(res) else 1)
