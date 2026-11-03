#!/usr/bin/env python3
"""PS-1: interval enclosure of C* (eq:fqe-p1-cstar; p1_sharp_coefficient.tex).

The exact equality spectra and fidelity are checked by PM-4; the wall value
G_w(pi/4) vanishes algebraically by the definition of C*.
"""
from mpmath import iv
iv.dps = 40
S3 = iv.sqrt(3)
B = 1+1/S3
ln2 = iv.log(2)
A0 = iv.log(3)/ln2/2


def d2L(L):
    return 2-iv.sqrt(B*(1+L/S3))


p = (2+iv.sqrt(2))/4
H2p = -(p*iv.log(p)+(1-p)*iv.log(1-p))/ln2
Cstar = (A0-(1+2*H2p)/3)/d2L(iv.sqrt(2))
assert Cstar.a > iv.mpf('0.19057') and Cstar.b < iv.mpf('0.19059')
assert Cstar.a > iv.mpf('0.190578947012726887') and Cstar.b < iv.mpf('0.190578947012726888')
print('[PASS] PS-1 C* enclosed in (0.190578947012726887, 0.190578947012726888); equality algebra is checked by PM-4')

