#!/usr/bin/env python3
"""LP-19 / GS-19: exact Rains dual and 19-qubit graph witness.

The strict n=13,19 conclusions additionally use the *published* exact
nonexistence of nonadditive ((13,1,6))_2 and ((19,1,8))_2 quantum codes:
Angles Munne--Huber, arXiv:2603.19901 (2026). This checker does NOT
reproduce their independent quantum-code nonexistence certificates.
"""
from fractions import Fraction as F
from itertools import combinations
from math import comb
from collections import Counter

N,K=19,9
L=F(93,38896)
E={0:F(-258,46189),1:F(107,4862),2:F(-2466,60775),7:F(-12,275)}
Y={0:F(1264751,18919014400),1:F(23,739024),2:F(9,707200),
   3:F(3,739024),4:F(18431,18919014400),8:F(179,3783802880),
   12:F(7,756760576),16:F(3,58212352)}
W={1:F(151,134368),2:F(33,67184),3:F(291,1478048),
   4:F(207,2956096),5:F(127,5912192),6:F(15,2956096),7:F(5,5912192)}
GRAPH19=('0de56','43c05','76a2b','55f74','41549','7558c','6d219',
         '76020','47038','0444d','6a23b','5500f','4c97b','4c586',
         '13bed','13441','4c8ac','404e4','33dfe')

def a(n,s,j):
    return F(comb(n-j,s-j),2**s*comb(n,s)) if j<=s else F(0)

def shadow(n,i,j):
    return F(sum(comb(n-j,u)*3**u*comb(j,i-u)*(-1)**(j-i+u)
                 for u in range(max(0,i-j),min(i,n-j)+1)),2**n)

def check_rains19():
    assert all(v>0 for v in [*Y.values(),*W.values()])
    for j in range(N+1):
        rhs=(L if j==0 else F(0))
        rhs+=sum(v*(a(N,s,j)-a(N,N-s,j)) for s,v in E.items())
        rhs+=sum(v*shadow(N,s,j) for s,v in Y.items())
        rhs+=W.get(j,F(0))
        assert rhs==a(N,K,j),(j,rhs,a(N,K,j))
    print('LP-19 PASS: 20 exact rational dual coefficients; pi_9 >=93/38896.')
    print('LP-19 PASS: equality forces all Pauli weights A1,...,A7 to vanish.')

def graph_rows():
    G=[int(s,16) for s in GRAPH19]
    assert len(G)==N
    assert all(g<(1<<N) and not (g>>i&1) for i,g in enumerate(G))
    assert all((G[i]>>j&1)==(G[j]>>i&1) for i in range(N) for j in range(N))
    return G

def check_graph():
    G=graph_rows()
    A=[0]*(N+1);z=prev=0
    for q in range(1<<N):
        x=q^(q>>1)
        if q:z^=G[(x^prev).bit_length()-1]
        prev=x
        A[(x|z).bit_count()]+=1
    assert sum(A)==1<<N
    v=sum(A[j]*comb(N-j,K-j) for j in range(K+1))
    assert v==122128,(v,A)
    assert F(v,2**K*comb(N,K))==F(449,173888)
    assert A[0]==1
    print('GS-19 PASS: exact stabilizer enumerator, normalized cut purity 449/173888')
    print('GS-19 full Pauli weight distribution:',A)
    def rank2(rows):
        piv={}
        for r in rows:
            while r:
                i=r.bit_length()-1
                if i not in piv:piv[i]=r;break
                r^=piv[i]
        return len(piv)
    hist=Counter()
    for S in combinations(range(N),K):
        mask=sum(1<<j for j in S)
        rank=rank2([G[i]&~mask for i in S])
        hist[rank]+=1
    assert sum(hist.values())==comb(N,K)
    purity=sum(F(cnt,2**r) for r,cnt in hist.items())/comb(N,K)
    assert purity==F(449,173888),(purity,hist)
    assert sorted(hist.items())==[(6,2),(7,975),(8,26811),(9,64590)]
    print('GS-19 PASS: all 92378 binary cut ranks, histogram',sorted(hist.items()))
    print('GS-19 upper bound: min pi_9(n=19) <= 449/173888.')

if __name__=='__main__':
    check_rains19();check_graph()
    print('EXTERNAL THEOREM (Angles Munne--Huber, arXiv:2603.19901):')
    print('Pure ((13,1,6))_2 and ((19,1,8))_2 quantum codes do not exist.')
    print('CONSEQUENCE: LP equality impossible at n=13 and n=19, so')
    print('25/1408 < min pi_6(13) <= 85/4576;')
    print('93/38896 < min pi_9(19) <= 449/173888.')
