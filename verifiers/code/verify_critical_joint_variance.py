#!/usr/bin/env python3
"""CJ: algebra in the analytic joint-cut critical variance proof.

Proved in: critical_quadratic_variance.tex. The correlation parametrisation,
polar Cauchy--Schwarz bound, unitary block estimates and sign/equality
arguments are proved in the text. No coefficient sign certificate is used.
"""
import sympy as s

Q = s.Rational
k, x, t, z = s.symbols('k x t z', real=True)
a = (4*k**3+3*k**2+2)/9
b = Q(2, 3)*(1-k)**2
c = Q(4, 9)*(1-k)**3
bk = a-b*x+c*x**Q(3, 2)
original = (k**2+Q(2, 3)*k*(1-k)**2*(1-x)
            +Q(2, 9)*(1-k)**3*(1-3*x+2*x**Q(3, 2)))
assert s.expand(bk-original) == 0
# Geometric concavity is a logarithmic second derivative, not coefficient signs.
numerator = s.expand((s.diff(bk, x)+x*s.diff(bk, x, 2))*bk
                     - x*s.diff(bk, x)**2)
target = -(4*a*b-9*a*c*s.sqrt(x)+b*c*x**Q(3, 2))/4
assert s.simplify(numerator-target) == 0
g = 4*a*b-9*a*c*t+b*c*t**3
assert s.expand(s.diff(g, t)-3*c*(b*t*t-3*a)) == 0
assert s.expand(a-b/3-k*(4*k*k+k+4)/9) == 0
assert s.factor(g.subs(t, 1)-4*k*k*(1-k)**2*(4*k*k+k+1)/9) == 0
f0 = k*k*(1-k)+Q(3, 2)*(1-k)**2*z-(1-k)**3*z**Q(3, 2)
assert s.expand(Q(1, 2)+Q(7, 4)*k*k-Q(9, 4)*bk.subs(x, z)-f0) == 0
print('CJ-1 PASS: directional permanent bound algebra and geometric-concavity identities')

# Three purities are one complex Fourier coefficient, with the exact factor 1/4.
w = s.symbols('w')
u = s.symbols('u0:3', real=True)
fourier = sum(w**j*u[j] for j in range(3))
fourier_bar = sum(w**(2*j)*u[j] for j in range(3))
norm_sq = s.rem(s.Poly(s.expand(fourier*fourier_bar), w),
                s.Poly(w*w+w+1, w)).as_expr()
uc = sum(u)/3
assert s.expand(norm_sq-Q(3, 2)*sum((j-uc)**2 for j in u)) == 0
purities = [(1+j)/4 for j in u]
P = sum(purities)
VP = sum((j-P/3)**2 for j in purities)
assert s.expand(norm_sq/4-6*VP) == 0
# The shared U,V blocks select the actual terms bounded in the text.
mm, nn = s.symbols('m0:6'), s.symbols('n0:6')
M = s.Matrix([[mm[0], mm[1], mm[2]], [mm[1], mm[3], mm[4]],
              [mm[2], mm[4], mm[5]]])
N = s.Matrix([[nn[0], nn[1], nn[2]], [nn[1], nn[3], nn[4]],
              [nn[2], nn[4], nn[5]]])
J = s.diag(k, -k, 1)
block = (mm[5]*nn[5]+2*k*(mm[2]*nn[2]-mm[4]*nn[4])
         +k*k*(mm[0]*nn[0]+mm[3]*nn[3]-2*mm[1]*nn[1]))
assert s.expand(s.trace(M*J*N*J)-block) == 0
X, Y = s.symbols('X Y', real=True)
assert s.expand((1-X)*(1-Y)-(1-X-Y+X*Y)) == 0
assert s.expand((1+X)*(1+Y)-(1+X+Y+X*Y)) == 0
assert s.expand(z+2*k*(1-z)+k*k*(1+z)-((1-k)**2*z+2*k+k*k)) == 0
print('CJ-2 PASS: Fourier variance factor and generic shared-unitary block expansion')

d = 1-k
f = d*k*k+Q(1, 2)*d*d*(1+2*k)*z+Q(1, 2)*d**3*z*(1-z)
B = d*d*z+2*k+k*k
R = -d**4*z**3+3*d**3*(1+k)*z*z-4*k*d*d*z+2*k*k*(1+2*k*k)
assert s.expand(4*f*f+9*k*k-B*B-3*k*k*(1-k*k)-d*d*(1-z)*R) == 0
lower = 2*d**3*(1+2*k)*z*z-4*k*d*d*z+2*k*k*(1+2*k*k)
assert s.expand(R-lower-d**4*z*z*(1-z)) == 0
disc = s.discriminant(lower, z)
assert s.factor(disc+16*k**3*d**3*(3+2*k+4*k*k)) == 0
f0_at_t = d*k*k+Q(3, 2)*d*d*t*t-d**3*t**3
assert s.expand(f0_at_t-f.subs(z, t*t)-Q(1, 2)*d**3*t*t*(1-t)**2) == 0
print('CJ-3 PASS: scalar factorisation, tangent bound and nonpositive discriminant identity')

# The complete cluster-to-Bell family; the critical formula is P=1+tau/2.
spectra = [[(1+k)/2, (1-k)/2, 0, 0], [Q(1, 4)]*4, [Q(1, 4)]*4]
p = [sum(y*y for y in row) for row in spectra]
P = sum(p)
T = sum(sum(y**3 for y in row) for row in spectra)
F = 17*P-12*T-12
V = sum((y-P/3)**2 for y in p)
assert s.expand(P-1-k*k/2) == 0
assert s.expand(F-(1-k*k)/2) == 0
assert s.expand(V-(1+2*k*k)**2/24) == 0
assert s.expand(F*F+3*P*(P-1)-6*V) == 0
assert s.expand(F*F+Q(9, 2)*(P-1)-6*V-Q(3, 4)*k*k*(1-k*k)) == 0
# diag(k,-k,1) gives these spectra using the same product-negative sign convention.
signs = [(1, 1, -1), (1, -1, 1), (-1, 1, 1), (-1, -1, -1)]
eigs = [(1+sig[0]*k-sig[1]*k+sig[2])/4 for sig in signs]
assert sorted(map(str, eigs)) == sorted(map(str, [0, 0, (1+k)/2, (1-k)/2]))
print('CJ-4 PASS: sharp family moments and tetrahedral spectra for all k')

# Dropping the shared correlation matrix fails, even retaining the critical P/tau
# identity and the sharp scalar critical envelope. This is not a physical triple.
bad = [[Q(55, 100), Q(25, 100), Q(10, 100), Q(10, 100)],
       [Q(1, 4)]*4, [Q(55, 100), Q(15, 100), Q(15, 100), Q(15, 100)]]
p = [sum(y*y for y in row) for row in bad]
P = sum(p)
F = 17*P-12*sum(sum(y**3 for y in row) for row in bad)-12
V = sum((y-P/3)**2 for y in p)
tau = Q(1, 100)
assert P == 1+tau/2 and F == tau-Q(1, 1000)
assert F*F+Q(9, 2)*(P-1)-6*V == -Q(43119, 10**6)
assert F*F+3*P*(P-1)-6*V < 0
print('CJ-5 PASS: scalar-only negative control; shared joint-cut structure is essential')
