#!/usr/bin/env python3
"""SC-2: exact rational lower witness E3(psi_{7/100}) > 59307/20000 = 2.96535.

Used for the entropy-contender restrictions (prop:seven-competitor-band,
cor:seven-entropy-eigen-cap-497). All checks are exact fractions with rational
arctanh logarithm enclosures.
"""
from fractions import Fraction as Q

def log_bounds(y, terms=10):
    assert y>0
    t=(y-1)/(y+1)
    p=t
    s=Q(0)
    for k in range(terms):
        s+=2*p/Q(2*k+1)
        p*=t*t
    rem=2*abs(p)/(Q(2*terms+1)*(1-t*t))
    return s-rem,s+rem

L2,U2=log_bounds(Q(2))

# An explicit attainable lower witness psi_{x=7/100}, with its four cut eigenvalues and multiplicities.
lam=(Q(43,375),Q(39,250),Q(86,375),Q(37,375))
weights=(195,40,15,30)
assert sum((w*t for w,t in zip(weights,lam)),Q(0))==35
loss_upper=sum((w*t*log_bounds(8*t)[1] for w,t in zip(weights,lam)),Q(0))/(35*L2)
assert loss_upper<Q(693,20000)

print('PASS SC-2: E3(psi_7/100) > 2.96535 = 59307/20000 by exact logarithm enclosures')
