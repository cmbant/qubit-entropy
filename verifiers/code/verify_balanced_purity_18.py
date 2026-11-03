#!/usr/bin/env python3
"""E18-1--E18-4: independent exact 18-qubit LP/graph-state certificates.

The graph is locally Clifford equivalent to an extremal Hermitian self-dual
GF(4) [18,9,8] code from the Harada--Munemasa database; the verifier is
self-contained and does not need an external code resource.

E18-1: rational Rains LP dual; E18-2: all 2^18 stabilizer words;
E18-3: all 48620 balanced cut ranks; E18-4: exact equality at 7/2288.
"""
from math import comb
from fractions import Fraction as F
from itertools import combinations
from collections import Counter

n=18; k=9; L=F(7,2288)
GRAPH='36e72 0fb7d 1b80a 0cad6 25e0b 30583 1c10b 2b228 1a262 3999b 27031 3221f 2a696 1dd87 2a45b 273ce 22b65 1deb1'
TARGET={0:1,8:2754,10:18360,12:77112,14:110160,16:50949,18:2808}
ES=list(map(F,['5/4004','-291/2002','81/364','278/1001',
               '-333/1001','0','-101/143','-54/143','51/26']))
YS={4:F('1/1166880'),6:F('1/1633632'),11:F('1/2498496')}
def choose(n,k):return comb(n,k) if 0<=k<=n else 0
def a(s,j):return F(choose(n-j,s-j),2**s*comb(n,s))
def shadow(i,j):
    return F(sum(choose(n-j,r)*3**r*choose(j,i-r)*(-1)**(j-i+r)
               for r in range(max(0,i-j),min(i,n-j)+1)),2**n)
def check_LP():
    w=[]
    for j in range(n+1):
        v=a(k,j)-(L if j==0 else 0)
        v-=sum(ES[s]*(a(s,j)-a(n-s,j)) for s in range(len(ES)))
        v-=sum(y*shadow(i,j) for i,y in YS.items())
        w.append(v)
    assert w[0]==0 and all(x>=0 for x in w),w
    print('E18-1 PASS: exact Rains dual pi_9>=7/2288; positive Pauli weights',
          [(i,str(v)) for i,v in enumerate(w) if v])
def rank_gf2(rows):
    basis={}
    for row in rows:
        while row:
            i=row.bit_length()-1
            if i in basis:row^=basis[i]
            else:basis[i]=row;break
    return len(basis)
def check_graph():
    G=[int(t,16) for t in GRAPH.split()]
    assert len(G)==n and all(not(G[i]>>i&1) for i in range(n))
    assert all((G[i]>>j&1)==(G[j]>>i&1) for i in range(n) for j in range(n))
    assert all(row.bit_count()%2 for row in G)
    H=Counter();x=z=0
    for t in range(1<<n):
        if t:
            i=(t&-t).bit_length()-1;x^=1<<i;z^=G[i]
        H[(x|z).bit_count()]+=1
    assert dict(H)==TARGET,dict(sorted(H.items()))
    print('E18-2 PASS: 2^18 stabilizer words, exact Type II distance-eight enumerator',dict(sorted(H.items())))
    ranks=Counter()
    for S in combinations(range(n),k):
        mask=sum(1<<i for i in S);other=((1<<n)-1)^mask
        ranks[rank_gf2([G[i]&other for i in S])]+=1
    assert dict(ranks)=={7:9180,9:39440},ranks
    print('E18-3 PASS: all 48620 9|9 ranks',dict(sorted(ranks.items())))
    p=sum(F(v,2**r) for r,v in ranks.items())/comb(n,k)
    q=sum(a(k,j)*v for j,v in H.items())
    assert p==q==L,(p,q,L)
    print('E18-4 PASS: exact graph purity',p,'= exact Rains bound')
if __name__=='__main__':
    check_LP();check_graph()
