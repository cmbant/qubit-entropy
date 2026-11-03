#!/usr/bin/env python3
"""EF-1--EF-7: exact checks for M4 mean Tsallis/Renyi entropy extensions.

Statement in spectral_corollaries.tex.
The all-real-q sign and Hermite majorization are proved there;
this verifier checks the symbolic coefficient and the cluster witness.
"""
import sympy as sp

q = sp.symbols("q", real=True)
p, r = sp.Rational(1, 6), sp.Rational(1, 2)
f = lambda t: (t**q - t)/(1-q)
fp = lambda t: (q*t**(q-1)-1)/(1-q)
h = r-p
a = (fp(p)+fp(r)-2*(f(r)-f(p))/h)/h**2
b = (fp(r)-fp(p))/(2*h)-sp.Rational(3,2)*a*(p+r)
target = 6**(3-q)/(48*(1-q))*(5+3*q-(15-7*q)*3**(q-1))
assert sp.simplify(sp.powsimp((b+sp.Rational(17,12)*a-target).expand(), force=True)) == 0
print("EF-1 PASS: general Hermite coefficient identity")

z = sp.zeros(16,1)
for bits,sign in (("0000",1),("0011",1),("1100",1),("1111",-1)):
    z[int(bits,2)] = sp.Rational(sign,2)
specs = []
for j in (1,2,3):
    A = sp.zeros(4)
    for i in range(16):
        bits = [int(v) for v in format(i,"04b")]
        rem = [k for k in (1,2,3) if k != j]
        A[2*bits[0]+bits[j],2*bits[rem[0]]+bits[rem[1]]] = z[i]
    ev = (A*A.H).eigenvals()
    specs.append(tuple(sorted([lam for lam,mult in ev.items() for _ in range(mult)])))
expected = sorted([(sp.Rational(0),sp.Rational(0),sp.Rational(1,2),sp.Rational(1,2)),
                   (sp.Rational(1,4),)*4,(sp.Rational(1,4),)*4])
assert sorted(specs) == expected
print("EF-2 PASS: cluster cut spectra")

mq = lambda v: sp.Rational(1,2)**v+3*sp.Rational(1,6)**v
cq = lambda v: (2*sp.Rational(1,4)**(v-1)+sp.Rational(1,2)**(v-1))/3
assert mq(2) == cq(2) == sp.Rational(1,3)
assert sp.simplify((mq(3)-cq(3))/2) == sp.Rational(1,144)
assert sp.simplify((mq(q)-cq(q))-
                   (2**(-q)/3)*(1+9*3**(-q)-8*2**(-q))) == 0
print("EF-3 PASS: q=2 tie, q=3 Tsallis cluster advantage 1/144, all-q moment identity")

assert sp.simplify(sp.limit(target,q,1)-
                   (6*sp.log(3)-sp.Rational(15,2))) == 0
assert sp.simplify(target.subs(q,2)) == -1
stability = 6*sp.log(2)/(sp.Rational(15,2)-6*sp.log(3))
assert 4.57862 < float(stability) < 4.57863
print("EF-4 PASS: von Neumann and q=2 limits; stability factor rounds to 4.57862")

# All-parameter derivatives; interval signs are proved in the paper.
x = sp.symbols("x", positive=True)
fq = (x**q-x)/(1-q)
assert sp.simplify(sp.diff(fq,x,3)-q*(2-q)*x**(q-3)) == 0
assert sp.simplify(sp.diff(fq,x,4)+q*(2-q)*(3-q)*x**(q-4)) == 0
R = (15-7*q)*3**(q-1)/(5+3*q)
assert sp.simplify(sp.diff(R,q)/R-sp.log(3)+80/((15-7*q)*(5+3*q))) == 0
assert sp.simplify(R.subs(q,0)-1) == sp.simplify(R.subs(q,1)-1) == 0
# ln(3) = sum_{k>=0} 1/((2k+1)4^k); first two terms and geometric tail.
assert sp.Rational(13,12) > sp.Rational(16,15)
assert 1+sp.Rational(1,3)*sp.Rational(1,3) == sp.Rational(10,9) < sp.Rational(5,4)
D = (15-7*q)*(5+3*q)
assert sp.expand(D-(75+10*q-21*q*q)) == 0 and D.subs(q,1) == 64
assert sp.diff(D,q).subs(q,sp.Rational(5,21)) == 0
print("EF-5 PASS: exact derivative and logarithmic-derivative identities, endpoints, rational log bounds and quadratic data")

# Stronger von Neumann defect coefficients from the polarization remainder.
a_entropy = 18*(sp.log(3)-1)
d_entropy = sp.Rational(15,2)-6*sp.log(3)
assert sp.simplify(d_entropy/2+a_entropy/16-(21-15*sp.log(3))/8) == 0   # F >= (3/4)W1 gives a/12 * 3/4 = a/16
assert sp.simplify(d_entropy/2-(sp.Rational(15,4)-3*sp.log(3))) == 0
# Envelope term (a/12)(tau - tau^{3/2}) of thm:fqe-general-envelope.
tau_old = sp.Rational(15, 4)-3*sp.log(3)
tau_new = tau_old+a_entropy/12
assert sp.simplify(a_entropy/12-sp.Rational(3, 2)*(sp.log(3)-1)) == 0
assert sp.simplify(tau_new-(9-6*sp.log(3))/4) == 0
assert round(float(tau_old), 3) == 0.454 and round(float(tau_new), 3) == 0.602
print("EF-6 PASS: stronger polarization and spin-flip entropy-defect coefficients; "
      "envelope tau coefficient (9-6 ln3)/4 = 0.602 vs 15/4-3 ln3 = 0.454")

# The universal defect (SF+) and the full-rank obstruction for mixed roofs.
ac, alpha, W1, tau, residual, pairs = sp.symbols('a alpha W1 tau residual pairs')
Pminus1 = (W1+tau)/2
Fc = sp.Rational(3, 4)*W1+sp.Rational(3, 8)*residual+sp.Rational(3, 2)*pairs   # identity (S) at N = 1
assert sp.expand(alpha*Pminus1+ac*Fc/12
                 -(alpha/2+ac/16)*W1-alpha*tau/2-ac*residual/32-ac*pairs/8) == 0
C = sp.Matrix(4,4,sp.symbols('c0:16'))
t = sp.symbols('t')
det_pencil = sp.Poly((sp.eye(4)+t*C).det(method='berkowitz'),t)
assert det_pencil.degree() == 4 and sp.expand(det_pencil.LC()-C.det(method='berkowitz')) == 0
print("EF-7 PASS: universal strengthened defect identity and determinant-pencil leading coefficient")
