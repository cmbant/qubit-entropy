#!/usr/bin/env python3
"""EU-1, EU-2: the sharp entropy--purity slope for every dimension and the entropy upper bounds of the
paper's table of other sizes (tab:fqe-other-n).

Proved in: lem:seven-entropy-linear-purity (seven_qubit_entropy_compatibility.tex, part of the supplement) and the
"Purity and entropy at other sizes" paragraph of four_qubit_entropy.tex.
EU-1 checks the symbolic identities of the one-spike proof for general m=d-1.
EU-2 evaluates k - c_d (d pi_min - 1), d=2^k, at 60 digits and checks each displayed 4-decimal upper bound.
     At n=13 the minimum is not known; the ceiling uses the lower bound min pi_6 >= 18171249/10^9 of
     prop:thirteeninterval (thm:threecopy-sw(a), SW-2--SW-4).  The ceiling decreases in pi, so a lower bound on
     the minimum purity gives a valid upper bound on the mean entropy.
"""
import sympy as s
from mpmath import mp, mpf, log, ceil

r, m = s.symbols('r m', positive=True)
F = (1 + m*r)*s.log(1 + m*r) + m*(1 - r)*s.log(1 - r)
N = r*s.diff(F, r) - 2*F
assert s.simplify(s.diff(N, r, 2) - m*(m + 1)*r*(2*m*r - (m - 1))/((1 - r)**2*(1 + m*r)**2)) == 0
rs = (m - 1)/m
assert s.simplify(s.expand_log(F.subs(r, rs) - (m - 1)*s.log(m), force=True)) == 0
assert s.simplify(s.expand_log(s.diff(F, r).subs(r, rs) - 2*m*s.log(m), force=True)) == 0
assert s.simplify(s.expand_log(N.subs(r, rs), force=True)) == 0
# minimum of F/r^2 divided by d*m*ln2 (d=m+1) is c_d=(d-1)log2(d-1)/(d(d-2))
d = m + 1
assert s.simplify((m**2*s.log(m)/(m - 1))/(d*m*s.log(2)) - (d - 1)*s.log(d - 1)/(s.log(2)*d*(d - 2))) == 0
for mm in range(2, 12):  # sign pattern of N'' on (0,1): negative below (m-1)/(2m), positive above
    for rr in (s.Rational(1, 1000), s.Rational(mm - 1, 2*mm) - s.Rational(1, 1000)):
        assert (2*mm*rr - (mm - 1)) < 0
    assert (2*mm*(s.Rational(mm - 1, 2*mm) + s.Rational(1, 1000)) - (mm - 1)) > 0
print('[PASS] EU-1 general one-spike identities: N\'\', F(r*), F\'(r*), N(r*)=0 and the slope c_d')

mp.dps = 60
TABLE = [  # n, k, min pi_k (supplement tab:minima; at n=13 a lower bound, prop:thirteeninterval), displayed bound
    (8, 4, mpf(3)/35, '3.9029'), (9, 4, mpf(1)/14, '3.9627'), (10, 5, mpf(1)/21, '4.9163'),
    (11, 5, mpf(1)/28, '4.9772'), (12, 6, mpf(5)/224, '5.9594'), (13, 6, mpf(18171249)/10**9, '5.9846'),
    (14, 7, mpf(331)/27456, '6.9702'),
    (15, 7, mpf(1)/104, '6.9874'), (16, 8, mpf(1)/156, '7.9800'), (17, 8, mpf(1)/208, '7.9928'),
    (18, 9, mpf(7)/2288, '8.9901')]
for n, k, p, shown in TABLE:
    dd = 2**k
    c = (dd - 1)*log(dd - 1, 2)/(dd*(dd - 2))
    bound = k - c*(dd*p - 1)
    up = ceil(bound*10**4)/10**4
    assert up == mpf(shown) and bound < up and up - bound < mpf(1)/10**4
    lab = 'pi_lower' if n == 13 else 'pi_min'
    print(f'[PASS] EU-2 n={n}: E_{k} <= {k} - c_{dd}({dd}*{lab}-1) = {mp.nstr(bound, 12)} < {shown}')
