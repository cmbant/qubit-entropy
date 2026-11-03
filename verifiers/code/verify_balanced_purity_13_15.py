#!/usr/bin/env python3
"""LP-13--LP-15, GS-14--GS-15, QP-13: exact balanced-purity certificates.

All bounds hold for unrestricted pure qubit states. Graph witnesses
attain the LP at n=14,15; no global n=13 optimum is asserted.
Integer/Fraction arithmetic only; no optimizer required.
"""
from collections import Counter
from fractions import Fraction as F
from itertools import combinations
from math import comb

DUALS={
 13: (F(25,1408),[-F(71,5148),F(1,44),F(0),-F(1,18),F(0),F(0),F(0)],
       {0:F(35,41184),1:F(1,3432),2:F(23,329472),6:F(1,164736),10:F(1,329472)},
       {1:F(445,54912),2:F(181,54912),3:F(21,18304),4:F(17,54912),5:F(1,18304)}),
 14: (F(331,27456),[-F(53,2002),F(35,858),F(0),F(0),-F(1,6),F(0),F(0)],
       {0:F(5,6864),1:F(7,18304),2:F(1,6864),3:F(89,2050048),7:F(1,186368),11:F(5,2050048)},
       {1:F(313,54912),2:F(1,416),3:F(49,54912),4:F(7,27456),5:F(1,18304)}),
 15: (F(1,104),[-F(128,6435),F(47,1287),F(0),F(0),F(0),-F(7,45),F(0),F(0)],
       {0:F(181,299520),1:F(17,51480),2:F(97,617760),3:F(1,17160),4:F(167,9884160),8:F(17,9884160),12:F(1,3294720)},
       {1:F(463,102960),2:F(19,10296),3:F(1,1430),4:F(1,5148),5:F(1,20592)})
}
# Vertex i has neighbours in the set bits of row i; rows in hexadecimal.
GRAPH14=('38e8','1bf0','0d68','3855','3a0a','0487','318f',
         '2563','2ac6','1512','32a4','011f','265b','15d9')
GRAPH15=('41d4','37bc','7dcb','6a96','580b','6742','4625','730f',
         '16a7','49ea','3966','061c','0596','04ae','02fd')

def arow(n,s):
    return [F(comb(n-j,s-j),2**s*comb(n,s)) if j<=s else F(0)
            for j in range(n+1)]

def shadow(n,i):
    return [F(sum(comb(n-j,u)*3**u*comb(j,i-u)*(-1)**(j-i+u)
                  for u in range(max(0,i-j),min(i,n-j)+1)),2**n)
            for j in range(n+1)]

def dot(row,A):
    return sum(x*y for x,y in zip(row,A))

def check_dual(n):
    L,e,y,w=DUALS[n]
    assert all(x>=0 for x in list(y.values())+list(w.values()))
    a=[arow(n,i) for i in range(n+1)]
    s=[shadow(n,i) for i in range(n+1)]
    assert len(e)==(n+1)//2
    for j in range(n+1):
        rhs=(L if j==0 else 0)+w.get(j,0)
        rhs+=sum(e[i]*(a[i][j]-a[n-i][j]) for i in range(len(e)))
        rhs+=sum(y[i]*s[i][j] for i in y)
        assert rhs==a[n//2][j],(n,j,rhs,a[n//2][j])
    print('LP-%d PASS: exact rational certificate pi_%d >= %s'%(n,n//2,L))

def graph_rows(hexrows):
    G=[int(x,16) for x in hexrows];n=len(G)
    assert all(row<(1<<n) and not row&(1<<i) for i,row in enumerate(G))
    assert all(bool(G[i]&(1<<j))==bool(G[j]&(1<<i))
               for i in range(n) for j in range(n))
    return G

def gray_stabilizers(G):
    """Tuples (selected X support, parity-Z support) for graph generators."""
    odd=prev=0
    for x in range(1<<len(G)):
        mask=x^(x>>1)
        if x: odd^=G[(mask^prev).bit_length()-1]
        prev=mask
        yield mask,odd

def weights(G):
    A=[0]*(len(G)+1)
    for x,z in gray_stabilizers(G): A[(x|z).bit_count()]+=1
    assert sum(A)==1<<len(G)
    return A

def rank2(rows):
    piv={}
    for r in rows:
        while r:
            i=r.bit_length()-1
            if i not in piv: piv[i]=r;break
            r^=piv[i]
    return len(piv)

def cut_hist(G,k):
    h=Counter()
    for S in combinations(range(len(G)),k):
        mask=sum(1<<j for j in S)
        h[rank2([G[i]&~mask for i in S])]+=1
    return dict(h)

def check_witness(n,hexrows,expected_A,expected_hist,edges):
    G=graph_rows(hexrows);A=weights(G);k=n//2
    assert sum(row.bit_count() for row in G)//2==edges
    assert A==expected_A,A
    hist=cut_hist(G,k)
    assert hist==expected_hist,(hist,expected_hist)
    pureA=dot(arow(n,k),A)
    pureCut=sum(F(c,2**r) for r,c in hist.items())/comb(n,k)
    assert pureA==pureCut==DUALS[n][0],(pureA,pureCut)
    assert all(dot(arow(n,s),A)==dot(arow(n,n-s),A) for s in range(n+1))
    print('GS-%d PASS: exact graph with %d edges, rank counts %s, purity %s' %
          (n,edges,dict(sorted(hist.items())),pureA))
    return G

def projection_weights(G,q,axis):
    """Pauli postselection on q; axes Z=1,X=2,Y=3."""
    m=len(G)-1
    counts=[[0]*(m+1) for _ in range(4)]
    for x,z in gray_stabilizers(G):
        pauli=2*((x>>q)&1)+((z>>q)&1)
        w=((x|z)&~(1<<q)).bit_count()
        counts[pauli][w]+=1
    assert counts[0][0]==1 and all(counts[a][0]==0 for a in (1,2,3))
    return [counts[0][j]+counts[axis][j] for j in range(m+1)]

def check_projection13(G):
    results=[]
    for q in range(len(G)):
        for axis in (1,2,3):
            A=projection_weights(G,q,axis)
            assert sum(A)==1<<13
            results.append((dot(arow(13,6),A),q,axis,A))
    assert min(x[0] for x in results)==F(85,4576)
    assert any(p==F(85,4576) and q==4 and axis==2 and A[5:7]==[15,204]
               for p,q,axis,A in results)
    print('QP-13 PASS: best postselection X on vertex 4, pi_6=85/4576.')
    print('QP-13: arbitrary pure-basis postselection is a convex combination')
    print('       of the X,Y,Z axis purity profiles (stabilizer restriction proof).')

if __name__=='__main__':
    for n in (13,14,15):check_dual(n)
    G14=check_witness(14,GRAPH14,
       [1,0,0,0,0,0,161,576,1113,2240,3738,4032,2870,1344,309],
       {5:60,6:1684,7:1688},47)
    check_witness(15,GRAPH15,
       [1,0,0,0,0,0,105,540,1305,2800,5418,7560,7350,5040,2205,444],
       {5:35,6:1380,7:5020},56)
    check_projection13(G14)
    print('EXACT: min pi_7(n=14)=331/27456, min pi_7(n=15)=1/104.')
    print('OPEN n=13: 25/1408 <= min pi_6 <= 85/4576.')
