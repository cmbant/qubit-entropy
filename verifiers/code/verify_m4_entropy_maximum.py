#!/usr/bin/env python3
"""ME-1--ME-6: the four-qubit average 2|2 entropy maximum (Higuchi--Sudbery state M4).

Statement in four_qubit_entropy.tex.
Exact arithmetic throughout. The 16 amplitudes z_x and their conjugates are independent
commuting indeterminates; a polynomial is a dict {(z-monomial, zbar-monomial): Fraction}.

ME-1  256 F = sum_{r,m,alpha} c_m (3|A+3B|^2 + 20|B|^2), c=(3,1,1,3), as a polynomial identity,
      where F = 17 n P - 12 T - 12 n^3 (all 3984 coefficients compared).
ME-2  the 2x2 Gram matrix of two one-spin-3/2 multiplicity contractions in five trace invariants, and
      32 F = 243 G_ss + 486 G_su + 423 G_uu with [[243,243],[243,423]] positive definite.
ME-3  purity identity  P = n^2 + (1/2)(sum_i (2 tr R_i^2 - n^2) + |z^T Y z|^2),  Y = sigma_y^{x4}.
ME-4  Hermite majorant of -x ln x at 1/6, 1/2 and the entropy inequality constants.
ME-5  F = 0 at M4 and at Bell(12)Bell(34); hence 17/12 is the optimal slope.
ME-6  the Cartan slice G_abcd: cut spectra |v|^2, |Hv|^2, |H'v|^2, spin-flip amplitude = sum v^2, Clifford
      realisations of W(D4) generators, and the equality solutions (two classes, M4 and conj M4).
"""
# log: verify_m4_entropy_maximum.txt
from collections import defaultdict
from fractions import Fraction as Fr
from itertools import combinations, product
import sympy as sp

# ---------- exact polynomials in z, zbar ----------
def clean(p):
    return {k: v for k, v in p.items() if v}

def pmul(p, q):
    out = defaultdict(Fr)
    for (a, b), u in p.items():
        for (c, d), v in q.items():
            out[(tuple(sorted(a + c)), tuple(sorted(b + d)))] += u * v
    return clean(out)

def padd(*terms):
    out = defaultdict(Fr)
    for w, p in terms:
        for k, v in p.items():
            out[k] += Fr(w) * v
    return clean(out)

def conj(p):
    return {(b, a): v for (a, b), v in p.items()}

def bits(x):
    return [(x >> (3 - k)) & 1 for k in range(4)]  # site 1 is the leading bit

def idx(bs):
    return sum(b << (3 - k) for k, b in enumerate(bs))

def rho(S):
    """Unnormalised reduced matrix R_S[a][a'] = sum_rest z_{a,rest} zbar_{a',rest}."""
    rest = [k for k in range(4) if k not in S]
    d = 2 ** len(S)
    R = [[defaultdict(Fr) for _ in range(d)] for _ in range(d)]
    for a, a2, c in product(range(d), range(d), range(2 ** len(rest))):
        x, y = [0] * 4, [0] * 4
        for i, k in enumerate(S):
            x[k], y[k] = (a >> (len(S) - 1 - i)) & 1, (a2 >> (len(S) - 1 - i)) & 1
        for i, k in enumerate(rest):
            x[k] = y[k] = (c >> (len(rest) - 1 - i)) & 1
        R[a][a2][((idx(x),), (idx(y),))] += 1
    return [[clean(e) for e in row] for row in R]

def trpow(R, k):
    d = len(R)
    out = {}
    for path in product(range(d), repeat=k):
        t = {((), ()): Fr(1)}
        for i in range(k):
            t = pmul(t, R[path[i]][path[(i + 1) % k]])
        out = padd((1, out), (1, t))
    return out

n = clean({((x,), (x,)): Fr(1) for x in range(16)})
cuts = [(0, j) for j in (1, 2, 3)]
Rc = {S: rho(S) for S in cuts}
R1 = {i: rho((i,)) for i in range(4)}
n3 = pmul(n, pmul(n, n))
nP = pmul(n, padd(*[(1, trpow(Rc[S], 2)) for S in cuts]))
T = padd(*[(1, trpow(Rc[S], 3)) for S in cuts])
nS1 = pmul(n, padd(*[(1, trpow(R1[i], 2)) for i in range(4)]))
Pi2 = {}
for i, j in combinations(range(4), 2):
    Rij = rho((i, j))
    for a, b, a2, b2 in product(range(2), repeat=4):
        t = pmul(Rij[2 * a + b][2 * a2 + b2], pmul(R1[i][a2][a], R1[j][b2][b]))
        Pi2 = padd((1, Pi2), (1, t))
F = padd((17, nP), (-12, T), (-12, n3))

# ---------- the cubics K_i(z) = sum z_x z_y ztilde_t prod_k r_{i_k}(x_k y_k t_k) ----------
# ztilde_t = (-1)^{|t|} zbar_{t xor 1111}. Per-site rows on (x,y,t): spin 3/2 (0-3), s (4,5), a (6,7).
ROWS = [{0: 1}, {1: 1, 2: 1, 4: 1}, {3: 1, 5: 1, 6: 1}, {7: 1},
        {1: 2, 2: -1, 4: -1}, {6: -2, 5: 1, 3: 1}, {2: 1, 4: -1}, {3: 1, 5: -1}]

def K(rows):
    out = defaultdict(Fr)
    for term in product(*(ROWS[r].items() for r in rows)):
        x = y = t = 0
        c = Fr(1)
        for key, w in term:
            x, y, t, c = 2 * x + (key >> 2), 2 * y + ((key >> 1) & 1), 2 * t + (key & 1), c * w
        out[(tuple(sorted((x, y))), (15 - t,))] += -c if bin(t).count('1') % 2 else c
    return clean(out)

def AB(r, m, alpha):
    other = [k for k in range(4) if k != r]
    rows = [None] * 4
    rows[r] = m
    for i, k in enumerate(other):
        rows[k] = 4 + alpha[i]
    A = K(rows)
    B = {}
    for D in combinations(range(3), 2):
        rr = list(rows)
        for i in D:
            rr[other[i]] = 6 + alpha[i]
        B = padd((1, B), (1, K(rr)))
    return A, B

sos, Gss, Gsu, Guu = {}, {}, {}, {}
for r, m, alpha in product(range(4), range(4), product(range(2), repeat=3)):
    A, B = AB(r, m, alpha)
    b = sp.binomial(3, m)
    c = 3 // b
    ApB = padd((1, A), (3, B))
    sos = padd((1, sos), (3 * c, pmul(ApB, conj(ApB))), (20 * c, pmul(B, conj(B))))
    # orthonormal components: <sss,w> = A/(6 sqrt(6b)), <u~,w> = B/(2 sqrt(6b))
    Gss = padd((1, Gss), (Fr(1, 216 * b), pmul(A, conj(A))))
    Gsu = padd((1, Gsu), (Fr(1, 144 * b), pmul(conj(A), B)), (Fr(1, 144 * b), pmul(A, conj(B))))
    Guu = padd((1, Guu), (Fr(1, 24 * b), pmul(B, conj(B))))

diff = padd((1, sos), (-256, F))
print('ME-1 monomials: F', len(F), 'SOS', len(sos), 'mismatches', len(diff))
assert not diff and len(F) == 3984

inv = [n3, nS1, nP, T, Pi2]
def comb(cs):
    return padd(*[(Fr(c), p) for c, p in zip(cs, inv)])
want = {'Gss': (Gss, comb([Fr(16, 81), Fr(-113, 324), Fr(17, 81), Fr(-4, 27), Fr(19, 81)])),
        'Gsu': (Gsu, comb([Fr(-8, 9), Fr(35, 36), Fr(-17, 27), Fr(4, 9), Fr(-11, 27)])),
        'Guu': (Guu, comb([0, Fr(-11, 12), Fr(17, 9), Fr(-4, 3), Fr(1, 3)]))}
for name, (g, formula) in want.items():
    assert not padd((1, g), (-1, formula)), name
assert not padd((243, Gss), (486, Gsu), (423, Guu), (-32, F))
M = sp.Matrix([[243, 243], [243, 423]])
assert M.det() == 243 * 180 and M[0, 0] > 0
nW1 = padd((2, nS1), (-4, n3))
# SP-1's local maps identify ||R||^2 = 3||9A+19B||^2/8.
Rnorm = padd((Fr(243,8), Gss), (Fr(513,4), Gsu), (Fr(1083,8), Guu))
assert not padd((32, F), (-15, nW1), (-3, Rnorm))
assert not padd((1, Gss), (Fr(2,3), Gsu), (Fr(1,9), Guu), (-Fr(8,81), nW1))
print('ME-2 full polynomial identity 32F = 15nW1 + 3||R||^2; local-moment contraction exact')

tau = {}
for x in range(16):
    tau = padd((1, tau), ((-1) ** bin(x).count('1'), {(tuple(sorted((x, 15 - x))), ()): Fr(1)}))
lhs = padd(*[(1, trpow(Rc[S], 2)) for S in cuts])
nn = pmul(n, n)
rhs = padd((1, nn), (1, padd(*[(1, trpow(R1[i], 2)) for i in range(4)])), (-2, nn),
           (Fr(1, 2), pmul(tau, conj(tau))))
assert not padd((1, lhs), (-1, rhs))
generic_A = sp.Matrix(2, 2, sp.symbols('a0:4'))
eps = sp.Matrix([[0,1],[-1,0]])
assert eps*generic_A.T*eps.T == sp.trace(generic_A)*sp.eye(2)-generic_A
print('ME-3 P = 1 + Delta/2, Delta = sum_i |r_i|^2 + |z^T Y z|^2; local spin-flip inversion: exact')

x, X = sp.symbols('x X', positive=True)
f = -x * sp.log(x)
a3, a2, a1, a0 = sp.symbols('a3 a2 a1 a0')
h = a3 * x**3 + a2 * x**2 + a1 * x + a0
sol = sp.solve([(h - f).subs(x, v) for v in (sp.Rational(1, 6), sp.Rational(1, 2))]
               + [sp.diff(h - f, x).subs(x, v) for v in (sp.Rational(1, 6), sp.Rational(1, 2))],
               [a3, a2, a1, a0], dict=True)[0]
L3 = sp.log(3)
assert sp.simplify(sol[a3] - 18 * (L3 - 1)) == 0 and sp.simplify(sol[a2] - (18 - sp.Rational(39, 2) * L3)) == 0
assert sp.simplify(sp.diff(f, x, 4) + 2 / x**3) == 0
lin = sp.simplify(sp.Rational(17, 12) * sol[a3] + sol[a2])
assert sp.simplify(lin - (6 * L3 - sp.Rational(15, 2))) == 0 and (sp.Rational(15, 2) - 6 * L3).evalf(30) > 0
assert sol[a3].evalf(30) > 0
smax = sp.simplify(-sol[a2] / sol[a3])
print('ME-4 a=18(ln3-1)>0, b=18-39ln3/2, 17a/12+b=6ln3-15/2<0; usable slope bound', sp.N(smax, 12))
Pv, Fv = sp.symbols('P F')
Tv = (17 * Pv - 12 - Fv) / 12
gap = (sol[a3] * (sp.Rational(5, 12) - Tv) + sol[a2] * (1 - Pv)) / (3 * sp.log(2))
target = (sp.Rational(15, 2) - 6 * L3) * (Pv - 1) / (3 * sp.log(2)) + (L3 - 1) * Fv / (2 * sp.log(2))
assert sp.simplify(gap - target) == 0
Em4 = (sp.Rational(1, 2) * sp.log(2) + 3 * sp.Rational(1, 6) * sp.log(6)) / sp.log(2)
assert sp.simplify(Em4 - (1 + sp.log(3) / (2 * sp.log(2)))) == 0
print('ME-4 E(M4) - E(psi) >= (15/2-6ln3)(P-1)/(3ln2) + (ln3-1)F/(2ln2); E(M4) = 1 + log2(3)/2')

w3 = (-1 + sp.I * sp.sqrt(3)) / 2  # exp(2 pi i/3)
def state(entries):
    z = [0] * 16
    for s, c in entries:
        z[int(s, 2)] += c
    return sp.Matrix(z)
def num_inv(z):
    zs = sp.Matrix(z)
    P_ = T_ = 0
    for j in (1, 2, 3):
        rest = [k for k in (1, 2, 3) if k != j]
        A = sp.zeros(4, 4)
        for xx in range(16):
            bb = bits(xx)
            A[2 * bb[0] + bb[j], 2 * bb[rest[0]] + bb[rest[1]]] = zs[xx]
        Rm = A * A.H
        P_, T_ = P_ + (Rm**2).trace(), T_ + (Rm**3).trace()
    return sp.nsimplify(sp.simplify(sp.expand(P_))), sp.nsimplify(sp.simplify(sp.expand(T_)))
M4 = state([('0011', 1), ('1100', 1), ('0101', w3), ('1010', w3), ('0110', w3**2), ('1001', w3**2)]) / sp.sqrt(6)
BB = state([('0000', 1), ('0011', 1), ('1100', 1), ('1111', 1)]) / 2
for name, z, PT in (('M4', M4, (1, sp.Rational(5, 12))), ('Bell(12)Bell(34)', BB, (sp.Rational(3, 2), sp.Rational(9, 8)))):
    Pz, Tz = num_inv(z)
    assert (Pz, Tz) == PT and 17 * Pz - 12 * Tz - 12 == 0
    print('ME-5', name, 'P =', Pz, 'T =', Tz, 'F = 0')
print('ME-5 so T <= 5/12 + s(P-1) holds for all states only with s >= 17/12 (Bell pair), used s =', sp.Rational(17, 12))

va, vb, vc, vd = sp.symbols('va vb vc vd')
def G(v):
    a, b, c, d = v
    return state([('0000', (a + d) / 2), ('1111', (a + d) / 2), ('0011', (a - d) / 2), ('1100', (a - d) / 2),
                  ('0101', (b + c) / 2), ('1010', (b + c) / 2), ('0110', (b - c) / 2), ('1001', (b - c) / 2)])
H1 = sp.Matrix([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]]) / 2
H2 = sp.Matrix([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]) / 2
re = sp.symbols('p0:4', real=True)
im = sp.symbols('q0:4', real=True)
v = sp.Matrix([re[k] + sp.I * im[k] for k in range(4)])
z = G(list(v))
lam = sp.Symbol('lam')
for j, Hm in zip((1, 2, 3), (sp.eye(4), H1, H2)):
    rest = [k for k in (1, 2, 3) if k != j]
    A = sp.zeros(4, 4)
    for xx in range(16):
        bb = bits(xx)
        A[2 * bb[0] + bb[j], 2 * bb[rest[0]] + bb[rest[1]]] = z[xx]
    wv = Hm * v
    cp_ = (A * A.H).charpoly(lam).as_expr()
    target = sp.prod([lam - sp.expand(wv[k] * sp.conjugate(wv[k])) for k in range(4)])
    assert sp.expand(cp_ - target) == 0
Yd = [(-1) ** bin(xx).count('1') for xx in range(16)]
tauG = sp.expand(sum(z[xx] * z[15 - xx] * Yd[xx] for xx in range(16)))
assert sp.expand(tauG - sum(v[k]**2 for k in range(4))) == 0
print('ME-6 G(v): cut spectra |v|^2, |H1 v|^2, |H2 v|^2 and spin-flip amplitude = sum v_k^2 (exact)')
Xp, Zp, Hd = sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[1, 0], [0, -1]]), sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
Sg, I2 = sp.Matrix([[1, 0], [0, sp.I]]), sp.eye(2)
def kron4(ops):
    out = ops[0]
    for o in ops[1:]:
        out = sp.kronecker_product(out, o)
    return out
sym = sp.Matrix([va, vb, vc, vd])
gens = {'(b d)': (kron4([Hd] * 4), [va, vd, vc, vb]),
        '(a d)': (kron4([Sg] * 4) * kron4([Zp, Zp, I2, I2]), [vd, vb, vc, va]),
        '(a b)(c d)': (kron4([Xp, I2, Xp, I2]), [vb, va, vd, vc]),
        '(c,d)->(-c,-d)': (kron4([I2, I2, Xp, Xp]), [va, vb, -vc, -vd])}
for name, (U, img) in gens.items():
    lhs_, rhs_ = sp.expand(U * G(list(sym))), sp.expand(G(img))
    ratio = [sp.simplify(lhs_[k] / rhs_[k]) for k in range(16) if rhs_[k] != 0]
    assert len(set(ratio)) == 1 and abs(complex(ratio[0])) == 1, name  # equal up to a global phase
print('ME-6 Clifford local unitaries realise generators of W(D4) on G(v):', ', '.join(gens))
vM4 = sp.Matrix([1, -1, sp.I * sp.sqrt(3), -1]) / sp.sqrt(6)
assert sp.simplify(G(list(vM4)) - M4) == sp.zeros(16, 1)
vperm = -sp.I * sp.Matrix([vM4[2], vM4[1], vM4[0], vM4[3]])  # global phase and the transposition (a c)
assert sp.simplify(vperm - sp.Matrix([sp.sqrt(3), sp.I, -sp.I, sp.I]) / sp.sqrt(6)) == sp.zeros(4, 1)
for eps in product((1, -1), repeat=3):
    vv = sp.Matrix([sp.sqrt(3)] + [e * sp.I for e in eps]) / sp.sqrt(6)
    assert sp.simplify(sum(t**2 for t in vv)) == 0
    for Hm in (sp.eye(4), H1, H2):
        spec = sorted(sp.nsimplify(sp.simplify(abs(t)**2)) for t in Hm * vv)
        assert spec == [sp.Rational(1, 6)] * 3 + [sp.Rational(1, 2)]
print('ME-6 the 8 solutions (sqrt3, +-i, +-i, +-i)/sqrt6 have tau = 0 and all spectra (1/2,1/6,1/6,1/6);',
      'M4 = G((1,-1,i sqrt3,-1)/sqrt6), a phase and W(D4) image of the class eps1 eps2 eps3 = -1')
print('ALL ME-1..ME-6 PASS')
