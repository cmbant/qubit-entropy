#!/usr/bin/env python3
"""TY-1--TY-2: exact Type II circulant graph states at n=16 and n=20.

C16(2,3,4,8) and C20(3,4,5,10) have odd degree at every vertex
(spinflip overlap tau=1), and exact balanced purity 1/143 and
131/77792. Together with ST-5/6 rational duals this proves sharp
spinflip-class optima and optimality of the tradeoff coefficients.
The code independently enumerates all stabilizers and all balanced cuts.
"""
from collections import Counter
from itertools import combinations
from fractions import Fraction as F
from math import comb
DATA=[
 (16,(2,3,4,8),F(1,143),
  {0:1,6:168,8:2610,10:14448,12:29400,14:17640,16:1269},
  {6:1096,7:6882,8:4892}),
 (20,(3,4,5,10),F(131,77792),
  {0:1,8:1710,10:20976,12:126540,14:355680,16:395865,18:141360,20:6444},
  {8:12312,9:96900,10:75544})
]
def rank2(rows):
    basis={}
    for v in rows:
        while v:
            i=v.bit_length()-1
            if i in basis:v^=basis[i]
            else:basis[i]=v;break
    return len(basis)
def run(n,ds,expected,weights,ranks):
    G=[0]*n
    for i in range(n):
        for d in ds:
            G[i]|=1<<((i+d)%n)
            G[i]|=1<<((i-d)%n)
    assert all(v.bit_count()%2 for v in G)
    H=Counter();x=z=0
    for t in range(1<<n):
        if t:
            j=(t&-t).bit_length()-1;x^=1<<j;z^=G[j]
        H[(x|z).bit_count()]+=1
    assert dict(H)==weights
    a=sum(F(v*comb(n-j,n//2-j),2**(n//2)*comb(n,n//2))
          for j,v in H.items() if j<=n//2)
    C=Counter()
    for S in combinations(range(n),n//2):
        mask=sum(1<<i for i in S);other=(1<<n)-1^mask
        C[rank2([G[i]&other for i in S])]+=1
    assert dict(C)==ranks
    b=sum(F(v,2**j) for j,v in C.items())/comb(n,n//2)
    assert a==b==expected,(n,a,b,expected)
    print('TY-%d PASS: C%d%s, tau=1, purity=%s, exact ranks %s'%(
          1 if n==16 else 2,n,ds,a,dict(sorted(C.items()))))
if __name__=='__main__':
    for row in DATA:run(*row)
