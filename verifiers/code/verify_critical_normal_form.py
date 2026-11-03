#!/usr/bin/env python3
"""Exact checks for critical_four_qubit_normal_form.tex.
CN-1: magic-basis generators, marginal identities and moment maps.
CN-2: G_abcd amplitudes.
CN-3: arbitrary-dimensional real-SVD block-elimination identities.
CN-4: rank-deficient rectangular and repeated-positive-block factorisations.
The arbitrary-rank existence argument is analytic in the paper.
"""
# log: verify_critical_normal_form.txt
import sympy as sp

res = []


def check(tag, ok, msg):
    res.append(ok)
    print(('[PASS] ' if ok else '[FAIL] ') + tag + ' ' + msg)


s2 = sp.sqrt(2)
m = [sp.Matrix([1, 0, 0, 1]) / s2, sp.I * sp.Matrix([0, 1, 1, 0]) / s2, sp.Matrix([0, 1, -1, 0]) / s2, sp.I * sp.Matrix([1, 0, 0, -1]) / s2]
Mg = sp.Matrix.hstack(*m)
assert (Mg.H * Mg).applyfunc(sp.simplify) == sp.eye(4)
sig = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]), sp.Matrix([[1, 0], [0, -1]])]
I2 = sp.eye(2)
gens = [sp.I * sp.kronecker_product(s, I2) for s in sig] + [sp.I * sp.kronecker_product(I2, s) for s in sig]
Lm = [(Mg.H * g * Mg).applyfunc(sp.simplify) for g in gens]
real_antisym = all(L.applyfunc(sp.im) == sp.zeros(4, 4) and (L + L.T) == sp.zeros(4, 4) for L in Lm)
# independence: stack the strict upper triangles
iu = [(i, j) for i in range(4) for j in range(i + 1, 4)]
Gm = sp.Matrix([[L[i, j] for (i, j) in iu] for L in Lm])
indep = Gm.rank() == 6

# symbolic Z, Bloch vectors as exact linear functions of Im(ZZ^dagger) / Im(Z^dagger Z)
xs = sp.symbols('x0:16', real=True)
ys = sp.symbols('y0:16', real=True)
Z = sp.Matrix(4, 4, lambda i, j: xs[4 * i + j] + sp.I * ys[4 * i + j])
psi = (Mg * Z * Mg.T).reshape(16, 1)


def marg(psi, qubits_keep):
    rest = [k for k in range(4) if k not in qubits_keep]
    d = 2 ** len(qubits_keep)
    M = sp.zeros(d, 2 ** len(rest))
    for x in range(16):
        b = [(x >> (3 - k)) & 1 for k in range(4)]
        r = sum(b[k] << (len(qubits_keep) - 1 - i) for i, k in enumerate(qubits_keep))
        c = sum(b[k] << (len(rest) - 1 - i) for i, k in enumerate(rest))
        M[r, c] += psi[x]
    return M * M.H


def bloch(rho1):
    return [sp.expand(sp.re(sp.expand((rho1 * s).trace()))) for s in sig]


r12 = marg(psi, [0, 1])
# check rho_12 = Z Z^dagger in the magic basis
rho12_magic = (Mg.H * r12 * Mg).applyfunc(sp.expand)
ok_rho = (rho12_magic - (Z * Z.H)).applyfunc(sp.expand) == sp.zeros(4, 4)
r34 = marg(psi, [2, 3])
rho34_magic = (Mg.H * r34 * Mg).applyfunc(sp.expand)
ok_rho &= (rho34_magic - (Z.H * Z).T).applyfunc(sp.expand) == sp.zeros(4, 4)
H12 = Z * Z.H
ImH12 = H12.applyfunc(lambda e: sp.expand(sp.im(sp.expand(e))))
H34 = (Z.H * Z)
ImH34 = H34.applyfunc(lambda e: sp.expand(sp.im(sp.expand(e))))
ok_lin = True
for rho, ImH, keep in ((r12, ImH12, (0, 1)), (r34, ImH34, (2, 3))):
    comps = []
    for q in keep:
        q_r = marg(psi, [q])
        comps += bloch(q_r)
    # the six components are linear in the six upper-triangle entries of Im H with a constant invertible matrix
    vecs = [ImH[i, j] for (i, j) in iu]
    cvars = list(xs) + list(ys)
    # the components are quadratic forms in (x,y); compare with Im H entries (also quadratic) by solving for a 6x6 map
    unknown = sp.symbols('a0:36')
    Mx = sp.Matrix(6, 6, unknown)
    diff = [sp.expand(comps[a] - sum(Mx[a, b] * vecs[b] for b in range(6))) for a in range(6)]
    eqs = []
    for dpoly in diff:
        poly = sp.Poly(dpoly, *cvars)
        eqs += poly.coeffs()
    sol = sp.solve(eqs, unknown, dict=True)
    ok_lin &= bool(sol) and all(len(s_) == 36 for s_ in sol)
    if sol:
        Msol = Mx.subs(sol[0])
        ok_lin &= Msol.rank() == 6
check('CN-1', bool(real_antisym and indep and ok_rho and ok_lin),
      'six generators real antisymmetric and independent in the magic basis; rho_12 = Z Z^dagger, rho_34 = (Z^dagger Z)^T; '
      'Bloch vectors of qubits 1,2 (3,4) are an exact invertible linear function of Im(Z Z^dagger) (Im(Z^dagger Z))')

# CN-2
v = sp.symbols('v1:5')
G = sum((v[k] * sp.kronecker_product(m[k], m[k]) for k in range(4)), sp.zeros(16, 1))
a, b, c, d = v[0], -v[1], v[2], -v[3]
Gabcd = sp.zeros(16, 1)
for st, val in (('0000', (a + d) / 2), ('1111', (a + d) / 2), ('0011', (a - d) / 2), ('1100', (a - d) / 2),
                ('0101', (b + c) / 2), ('1010', (b + c) / 2), ('0110', (b - c) / 2), ('1001', (b - c) / 2)):
    Gabcd[int(st, 2)] += val
check('CN-2', (G - Gabcd).applyfunc(sp.expand) == sp.zeros(16, 1), 'sum_k v_k m_k x m_k = G_abcd with (a,b,c,d) = (v1,-v2,v3,-v4)')

# CN-3: eliminate a pair of entries using the two Gram symmetry equations.
d, e, u, v = sp.symbols('d e u v', real=True)
E1, E2 = d*v-e*u, d*u-e*v
check('CN-3', sp.expand(d*E2+e*E1-(d*d-e*e)*u) == 0
      and sp.expand(e*E2+d*E1-(d*d-e*e)*v) == 0
      and sp.expand(E1.subs(e, d)-d*(v-u)) == 0
      and E1.subs(e, 0) == d*v and E2.subs(e, 0) == d*u,
      'two real Gram conditions eliminate unequal singular-value blocks; '
      'equal positive blocks are symmetric; positive/null couplings vanish')
# CN-4: a repeated positive singular-value block and a rectangular nullspace.
R = sp.Matrix([[2, -1], [1, 2]]) / sp.sqrt(5)
N = sp.Matrix([[3, -4], [4, 3]]) / 5
L, Q = sp.diag(R, 1), sp.diag(R, N)
X = sp.Matrix([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0]])
Y = sp.Matrix([[3, 4, 0, 0], [4, -3, 0, 0], [0, 0, 3, 4]])
Z = X + sp.I*Y
D = sp.Matrix([[2+5*sp.I, 0, 0, 0], [0, 2-5*sp.I, 0, 0], [0, 0, 5*sp.I, 0]])
zero = lambda M: M.applyfunc(sp.simplify) == sp.zeros(*M.shape)
ok4 = zero(L.T*L-sp.eye(3)) and zero(Q.T*Q-sp.eye(4))
ok4 &= zero(Z-L*D*Q.T) and zero(Z*Z.H-(Z*Z.H).conjugate())
ok4 &= zero(Z.H*Z-(Z.H*Z).conjugate())
ok4 &= zero(Z.T-Q*D.T*L.T)  # also m > n
# All-real-part-zero case: independent left/right SVD on the entire matrix.
D0 = sp.I*sp.Matrix([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 3, 0]])
Z0 = L*D0*Q.T
ok4 &= zero(Z0.conjugate()+Z0) and zero(Z0*Z0.H-(Z0*Z0.H).conjugate())
ok4 &= zero(Z0.H*Z0-(Z0.H*Z0).conjugate())
check('CN-4', ok4, 'exact rectangular examples: repeated positive block, '
      'unequal nullspaces, rank deficiency, transpose and zero real part')
print('All CN checks passed' if all(res) else 'SOME CN CHECK FAILED')
raise SystemExit(0 if all(res) else 1)
