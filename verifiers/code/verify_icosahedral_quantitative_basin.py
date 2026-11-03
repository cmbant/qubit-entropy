#!/usr/bin/env python3
"""QB-1--QB-3: exact certified 7q orbital basin and flat-phase equivalence.

Depends on LH-1--LH-4 (certify_icosahedral_full_hessian_exact.py):
real Lagrange Hessian transverse to LU has quadratic-form bound h>78/4585.
Analytically, ||G||<=1 for the mean two-copy swaps, so a quartic expansion
about the closest orbit vector has remainder |R3+R4|<=4*d^3+d^4.
"""
from fractions import Fraction as F
from itertools import combinations

h=F(78,4585); radius=F(1,250); c=F(1,1024)
margin=h-4*radius-radius*radius-c
assert margin==F(285123,14672000000)>0
print('QB-1 PASS: exact rational quartic remainder margin:',margin)
print('QB-1: for LU-orbit distance d<=1/250, pi_3>=121/917 + d^2/1024.')

assert radius*radius/F(2)==F(1,125000)
print('QB-2 PASS: any inequivalent state of purity <=121/917 has')
print('      d>1/250 and max LU overlap amplitude <124999/125000.')

A={(0,2),(1,0)}
B={(0,6),(1,4)}
assert all(2-s-F(w,2)==1 for s,w in A)
assert all(2-s-F(w,2)==-1 for s,w in B)
pent={(1,3),(2,3),(2,4),(4,5),(1,5)}
support={}
for i,j in combinations(range(6),2):
    sg=-1 if (i,j) in pent else 1
    x=[0]*7; x[1+i]=x[1+j]=1
    support[tuple(x)]=('A',sg)
    y=[1]*7; y[1+i]=y[1+j]=0
    support[tuple(y)]=('B',-sg)
support[(0,1,1,1,1,1,1)]=('B',1)
support[(1,0,0,0,0,0,0)]=('A',1)
assert len(support)==32
for bits,(sector,sgn) in support.items():
    sw=(bits[0],sum(bits[1:]))
    assert sw in (A if sector=='A' else B)
    assert 2-sw[0]-F(sw[1],2)==(1 if sector=='A' else -1)
print('QB-3 PASS: 32/32 nonzero amplitudes have the exact diagonal LU phase.')
print('NOTE: local basin only; global minimality is the five-copy certificate (cor:seven-global-from-fivecopy).')
