#!/usr/bin/env python3
"""NG-1--NG-3: exact improved 19-qubit graph witness from 3 edge toggles.

The starting graph is GS-19 in verify_balanced_purity_19.py. The new
state is a genuine stabilizer/graph state, not a moment-relaxation point.
Integer stabilizer support and cut ranks are checked independently.
"""
from collections import Counter
from fractions import Fraction as F
from itertools import combinations
from math import comb

N,K=19,9
GRAPH19=('0de56','43c05','76a2b','55f74','41549','7558c','6d219',
         '76020','47038','0444d','6a23b','5500f','4c97b','4c586',
         '13bed','13441','4c8ac','404e4','33dfe')
TOGGLES=((2,6),(2,16),(2,18))
G=[int(s,16) for s in GRAPH19]
for i,j in TOGGLES:
    G[i]^=1<<j
    G[j]^=1<<i
assert all(g<(1<<N) and not (g>>i&1) for i,g in enumerate(G))
assert all((G[i]>>j&1)==(G[j]>>i&1) for i in range(N) for j in range(N))
print('NG-1 PASS: three graph-edge toggles (2,6), (2,16), (2,18); loop-free symmetric adjacency.')

A=[0]*(N+1);z=0;prev=0
for q in range(1<<N):
    x=q^(q>>1)
    if q:z^=G[(x^prev).bit_length()-1]
    prev=x
    A[(x|z).bit_count()]+=1
assert sum(A)==1<<N
assert A==[1,0,0,0,0,0,3,195,1080,3622,10173,25173,51276,82932,
           106029,105789,79515,42246,14067,2187]
num=sum(A[j]*comb(N-j,K-j) for j in range(K+1))
assert num==121608
p=F(num,2**K*comb(N,K))
assert p==F(15201,5912192)<F(449,173888)
print('NG-2 PASS: exact 2^19 stabilizer distribution; purity numerator 121608; pi_9=15201/5912192.')

def rank2(rows):
    piv={}
    for v in rows:
        while v:
            i=v.bit_length()-1
            if i in piv:v^=piv[i]
            else:piv[i]=v;break
    return len(piv)

hist=Counter()
for S in combinations(range(N),K):
    mask=sum(1<<j for j in S)
    hist[rank2([G[i]&~mask for i in S])]+=1
assert dict(hist)=={6:2,7:840,8:26696,9:64840}
assert sum(hist.values())==comb(N,K)==92378
cut_p=sum((F(cnt,1<<r) for r,cnt in hist.items()),F(0))/comb(N,K)
assert cut_p==p
assert F(449,173888)-p==F(5,454784)
print('NG-3 PASS: all 92378 balanced cut ranks exactly {6:2, 7:840, 8:26696, 9:64840}.')
print('NG-3 PASS: independent cut purity equals 15201/5912192; improvement over GS-19 is 5/454784.')
print('NEW UPPER BOUND: min pi_9(19) <= 15201/5912192; global optimality unknown.')
