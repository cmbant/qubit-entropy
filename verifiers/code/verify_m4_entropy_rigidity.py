#!/usr/bin/env python3
"""ER-1--ER-5: exact sextic-zero, one-cut rigidity and entropy checks.
The previous tags GD-6, GD-7, GD-8 are ER-1, ER-2, ER-3 respectively.
The sextic is now derived independently in verify_m4_entropy_swap_proof.py.
ER-4 compares the critical-slice factorization with all direct partial traces;
ER-5 checks all zero representatives. The exhaustive classification is analytic
in sextic_identity.tex, not inferred from enumerating these representatives.
"""
from itertools import combinations
import sympy as sp
res = []
w3 = (-1 + sp.I*sp.sqrt(3))/2


def check(tag, ok, message):
    res.append(bool(ok))
    print(('[PASS] ' if ok else '[FAIL] ')+tag+' '+message)


def bits(x):
    return [(x >> k) & 1 for k in (3,2,1,0)]

def Gslice(v):
    a, b, c, d = v
    out = [0] * 16
    for s, val in (('0000', (a + d) / 2), ('1111', (a + d) / 2), ('0011', (a - d) / 2), ('1100', (a - d) / 2),
                   ('0101', (b + c) / 2), ('1010', (b + c) / 2), ('0110', (b - c) / 2), ('1001', (b - c) / 2)):
        out[int(s, 2)] += val
    return sp.Matrix(out)

def cut_matrix(z, j):
    rest = [k for k in (1, 2, 3) if k != j]
    A = sp.zeros(4, 4)
    for x in range(16):
        bb = bits(x)
        A[2 * bb[0] + bb[j], 2 * bb[rest[0]] + bb[rest[1]]] = z[x]
    return A

vs = sp.symbols('a b c d')
zs = Gslice(vs)
a_, b_, c_, d_ = vs
ok6 = True
# cut 12|34 is O diag O^T with O real orthogonal (Takagi): eigenvectors (1,1),(1,-1) on {00,11} and {01,10}
Oo = sp.Matrix([[1, 1, 0, 0], [1, -1, 0, 0], [0, 0, 1, 1], [0, 0, 1, -1]]) / sp.sqrt(2)
A12 = cut_matrix(zs, 1)
perm_idx = [0, 3, 1, 2]       # rows 00,01,10,11 -> blocks {00,11},{01,10}
P4 = sp.zeros(4, 4)
for i, k in enumerate(perm_idx):
    P4[i, k] = 1
ok6 &= sp.expand(P4 * A12 * P4.T - Oo * sp.diag(a_, d_, b_, c_) * Oo.T) == sp.zeros(4, 4)
ok6 &= sp.expand(sum((-1) ** bin(x).count('1') * zs[x] * zs[15 ^ x] for x in range(16)) - sum(t ** 2 for t in vs)) == 0

def act(ops, z):
    out = [0] * 16
    for x in range(16):
        bb = bits(x)
        sign = 1
        nb = list(bb)
        for k, o in enumerate(ops):
            if o == 'X':
                nb[k] ^= 1
            if o == 'Z' and bb[k]:
                sign = -sign
        out[idx_bits(nb)] += sign * z[x]
    return sp.Matrix(out)

def idx_bits(bs):
    return sum(bit << (3 - k) for k, bit in enumerate(bs))

sa, sb, sc, sd = sp.symbols('sa sb sc sd')
zz = Gslice([sa, sb, sc, sd])
acts = {('X', 'I', 'X', 'I'): [sb, sa, sd, sc], ('Z', 'I', 'Z', 'I'): [sd, sc, sb, sa],
        ('Z', 'Z', 'I', 'I'): [sa, -sb, -sc, sd], ('I', 'I', 'X', 'X'): [sa, sb, -sc, -sd]}
for ops, img in acts.items():
    ok6 &= sp.expand(act(list(ops), zz) - Gslice(img)) == sp.zeros(16, 1)
vM4 = sp.Matrix([1, -1, sp.I * sp.sqrt(3), -1]) / sp.sqrt(6)
M4z = sp.zeros(16,1)
for word, value in [('0011',1),('1100',1),('0101',w3),('1010',w3),('0110',w3**2),('1001',w3**2)]:
    M4z[int(word,2)] = value
ok6 &= sp.simplify(Gslice(list(vM4)) * sp.sqrt(6) - M4z) == sp.zeros(16, 1)
# phase -i, then (ac)(bd): (sqrt3, i, -i, i)/sqrt6
vv = -sp.I * vM4
vv = sp.Matrix([vv[2], vv[3], vv[0], vv[1]])
ok6 &= sp.simplify(vv - sp.Matrix([sp.sqrt(3), sp.I, -sp.I, sp.I]) / sp.sqrt(6)) == sp.zeros(4, 1)
check('ER-1', bool(ok6), 'one-cut spectrum |v|^2 and spin-flip amplitude = sum v_k^2 on G(v) (real Hadamard diagonalisation); '
      'X1X3, Z1Z3, Z1Z2, X3X4 act as (ab)(cd), (ad)(bc), flip(b,c), flip(c,d); M4 = G((1,-1,i sqrt3,-1)/sqrt6) '
      '~ (sqrt3, i, -i, i)/sqrt6')

# ER-2: the weighted triangle gap, with all circle relations visible.
r = sp.symbols('r0:3', real=True)
c = sp.symbols('c0:3', real=True)
t = sp.symbols('s0:3', real=True)
R = sum(r)
gap = R**2 - sum(r[i]*c[i] for i in range(3))**2 - sum(r[i]*t[i] for i in range(3))**2
pairs = sum(r[i]*r[j]*((c[i]-c[j])**2 + (t[i]-t[j])**2) for i, j in combinations(range(3), 2))
circles = R*sum(r[i]*(1-c[i]**2-t[i]**2) for i in range(3))
check('ER-2', sp.expand(gap-pairs-circles) == 0,
      'exact weighted triangle-deficit identity: at positive weights, equality forces all unit phases equal')

# ---------------------------------------------------------------- ER-3: the entropy identity with the Hermite remainder

x_ = sp.symbols('x', positive=True)
f_ = -x_ * sp.log(x_)
sol = sp.solve([(sum(c * x_ ** k for c, k in zip(sp.symbols('h0:4'), range(4))) - f_).subs(x_, v) for v in (sp.Rational(1, 6), sp.Rational(1, 2))]
               + [sp.diff(sum(c * x_ ** k for c, k in zip(sp.symbols('h0:4'), range(4))) - f_, x_).subs(x_, v) for v in (sp.Rational(1, 6), sp.Rational(1, 2))],
               sp.symbols('h0:4'), dict=True)[0]
h0, h1, h2, h3 = sp.symbols('h0:4')
a_c, b_c = sol[h3], sol[h2]
L3 = sp.log(3)
# leading coefficients of the cubic Hermite interpolant of -x log x at 1/6 and 1/2
ok8 = sp.simplify(a_c - 18 * (L3 - 1)) == 0 and sp.simplify(b_c - (18 - sp.Rational(39, 2) * L3)) == 0
Pv, Fv = sp.symbols('P F')
Tv = (17 * Pv - 12 - Fv) / 12
lhs = -(b_c * (Pv - 1) + a_c * (Tv - sp.Rational(5, 12)))
rhs = (sp.Rational(15, 2) - 6 * L3) * (Pv - 1) + a_c / 12 * Fv
ok8 &= sp.simplify(lhs - rhs) == 0 and sp.simplify(a_c / 12 - sp.Rational(3, 2) * (L3 - 1)) == 0
# remainder sign: f - h = f(xi)/24 (x-1/6)^2 (x-1/2)^2 with f = -2/x^3 < 0
ok8 &= sp.simplify(sp.diff(f_, x_, 4) + 2 / x_ ** 3) == 0
# E(M4): spectrum (1/2,1/6,1/6,1/6) on every cut
EM4 = sp.simplify((sp.Rational(1, 2) * sp.log(2) + 3 * sp.Rational(1, 6) * sp.log(6)) / sp.log(2))
ok8 &= sp.simplify(EM4 - (1 + L3 / (2 * sp.log(2)))) == 0
# 4 eigenvalues in {1/6,1/2} summing to 1: exactly one 1/2
ok8 &= [k for k in range(5) if sp.Rational(k, 2) + sp.Rational(4 - k, 6) == 1] == [1]
check('ER-3', bool(ok8), '3 ln2 (E(M4)-E) = (15/2 - 6 ln3)(P-1) + (3/2)(ln3-1) F + sum_{cuts,eigenvalues}(h-f) with h-f>=0; '
      'a = 18(ln3-1), b = 18 - 39 ln3/2; the only spectrum in {1/6,1/2} with trace 1 is (1/2,1/6,1/6,1/6)')

# ER-4: independent polynomial comparison using direct flattenings, not the
# Hadamard/sign-average formula used in the proof. v and vbar are independent.
v = sp.symbols('v0:4')
vb = sp.symbols('w0:4')
zv, zb = Gslice(v), Gslice(vb)
n = sum(v[i]*vb[i] for i in range(4))
pv = [v[i]*vb[i] for i in range(4)]
P, T = 0, 0
for cut in (1, 2, 3):
    Rcut = cut_matrix(zv, cut)*cut_matrix(zb, cut).T
    P += sp.trace(Rcut**2)
    T += sp.trace(Rcut**3)
Q = 0
for i in range(4):
    for j, k in combinations([a for a in range(4) if a != i], 2):
        qv = vb[i]*(v[j]**2-v[k]**2)+2*v[i]*(pv[j]-pv[k])
        qb = v[i]*(vb[j]**2-vb[k]**2)+2*vb[i]*(pv[j]-pv[k])
        Q += qv*qb
sv, sb = sum(a*a for a in v), sum(a*a for a in vb)
Bv = sum(pv[i]*v[i]**2 for i in range(4))
Bb = sum(pv[i]*vb[i]**2 for i in range(4))
Tformula = (sp.Rational(3,4)*n**3+sp.Rational(9,8)*n*sv*sb
            -sp.Rational(9,4)*n*sum(a*a for a in pv)
            +3*sum(a**3 for a in pv)-sp.Rational(3,4)*(sb*Bv+sv*Bb))
ok4 = sp.expand(P-n*n-sv*sb/2) == 0
ok4 &= sp.expand(T-Tformula) == 0
ok4 &= sp.expand(17*n*P-12*T-12*n**3-Q) == 0
check('ER-4', ok4, 'critical-slice twelve-term factorization and sign-average moment formulas; '
      'all coefficients compared against direct reduced matrices')

# ER-5: converse zero families, including all Bell-pair pairings. The proof
# excludes every other zero by the phase equations (ZE).
from itertools import product
Hplus = sp.Matrix([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]])/2
Hminus = sp.Matrix([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]])/2
Fpoly = sp.Poly(sp.expand(17*n*P-12*T-12*n**3), *v, *vb)


def exact_zero_at(w):
    substitution = dict(zip(v, w)) | dict(zip(vb, map(sp.conjugate, w)))
    return sp.simplify(Fpoly.as_expr().subs(substitution, simultaneous=True)) == 0


ok5 = True
for signs in product((-1,1), repeat=4):
    w = sp.Matrix(signs)/2
    images = [Hplus*w, Hminus*w]
    ok5 &= any(sum(a != 0 for a in image) == 1 for image in images)
    ok5 &= exact_zero_at(w)
for position in range(4):
    w = sp.eye(4)[:, position]
    ok5 &= exact_zero_at(w)
for signs in product((-1,1), repeat=3):
    w = sp.Matrix([sp.sqrt(3)]+[sp.I*a for a in signs])/sp.sqrt(6)
    ok5 &= exact_zero_at(w)
    ok5 &= sp.simplify(sum(a*a for a in w)) == 0
D12, D13 = [sp.simplify(cut_matrix(M4z/sp.sqrt(6),j).det()) for j in (1,2)]
ok5 &= sp.simplify(D13/D12-(1-sp.I*sp.sqrt(3))/2) == 0
check('ER-5', ok5, 'all HS sign representatives, coordinate Bell products and both '
      'Hadamard Bell-product sign families satisfy the sextic equality; '
      'D13/D12 at M4 = (1-i sqrt3)/2 distinguishes its conjugate')

print('All ER checks passed' if all(res) else 'SOME ER CHECK FAILED')
raise SystemExit(0 if all(res) else 1)
