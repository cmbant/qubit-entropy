#!/usr/bin/env python3
"""ST-1--ST-6: exact global purity--time-reversal tradeoff certificates.

For every normalized even-n pure qubit state:
 pi_{n/2}(psi) >= L_n + c_n |<psi|J_n psi>|^2.
Rational identities use Pauli-weight positivity, pure-state complementary
purities and Rains shadow positivity. No LP solver or floating arithmetic.
c_n is optimal for the stated L_n: a graph state with all degrees odd
(tau=1) has pi_{n/2}=L_n+c_n.  At n=4,8,10,14 it is the parity
purification of an (n-1)-vertex graph; at n=16,20 it is the circulant
C16(2,3,4,8) or C20(3,4,5,10) of TY-1/TY-2, whose purity is recomputed
here from its Pauli weight distribution.
"""
from math import comb
from fractions import Fraction as F

def choose(n,k):return comb(n,k) if 0<=k<=n else 0
def a_coeff(n,s,j):return F(choose(n-j,s-j),2**s*comb(n,s))
def shadow_coeff(n,i,j):
    return F(sum(choose(n-j,r)*3**r*choose(j,i-r)*(-1)**(j-i+r)
        for r in range(max(0,i-j),min(i,n-j)+1)),2**n)
def parity_obj(m,j):
    if j%2:return F(0)
    n=m+1;k=n//2
    return sum(F(choose(m-j,s-j),2**s*comb(n,k)) for s in (k,n-k))
def graph_weight(A):
    m=len(A);w=[0]*(m+1)
    for x in range(1<<m):
        y=0
        for i in range(m):
            if x>>i&1:y^=A[i]
        w[(x|y).bit_count()]+=1
    return w

CASES={
 4:('1/3','1/6',['1/6','-2/3'],{}),
 8:('3/35','1/70',['-1/14','0','0','0'],{1:'1/224',5:'1/1120'}),
 10:('1/21','1/168',['-3/56','0','0','0','-5/24'],
     {1:'11/2688',2:'1/504',3:'47/64512',7:'11/64512'}),
 14:('331/27456','5/6864',['-10/429','175/3432','0','0','0','-7/24','0'],
     {1:'5/13728',2:'1/6864',3:'5/109824',7:'1/219648'}),
 16:('1/156','1/1716',['-133/5070','992/27885','0','0','0','-448/2145','0','0'],
     {1:'18263/57108480',2:'17/102960',3:'139/1784640',4:'1/34320',
      5:'1489/171325440',9:'21/19036160',13:'7/11421696'}),
 20:('1193/739024','103/1478048',
     ['54/4199','-700/4199','4725/4862','-8100/2431','1050/143',
      '-1512/143','1350/143','-600/143','0','0'],
     {2:'23/1478048',4:'3/1478048'}),
}
SHARP_WITNESS={
 4:None,8:None,10:None,
 14:'1850 15ac d62 1412 1e69 c96 915 1e22 1a46 590 2be 11f5 99b',
 16:(2,3,4,8),20:(3,4,5,10)
}
def circulant_purity(n,ds):
    G=[0]*n
    for i in range(n):
        for d in ds:G[i]|=1<<((i+d)%n)|1<<((i-d)%n)
    assert all(v.bit_count()%2 for v in G)
    A=[0]*(n+1);x=z=0
    for t in range(1,1<<n):
        A[(x|z).bit_count()]+=1
        j=(t&-t).bit_length()-1;x^=1<<j;z^=G[j]
    A[(x|z).bit_count()]+=1
    assert A[0]==1 and sum(A)==2**n
    return sum(a_coeff(n,n//2,j)*A[j] for j in range(n//2+1))
def witness_base(n):
    if SHARP_WITNESS[n] is not None:
        return [int(x,16) for x in SHARP_WITNESS[n].split()]
    m=n-1
    return [1<<((i-1)%m)|1<<((i+1)%m) for i in range(m)]
def check(n,L,coef,es,ys):
    L=F(L);c=F(coef);es=[F(v) for v in es];ys={int(i):F(v) for i,v in ys.items()}
    assert len(es)==(n+1)//2 and all(v>=0 for v in ys.values())
    k=n//2;w=[]
    for j in range(n+1):
        v=a_coeff(n,k,j)-c*F((-1)**j,2**n)-(L if j==0 else 0)
        v-=sum(es[s]*(a_coeff(n,s,j)-a_coeff(n,n-s,j)) for s in range(len(es)))
        v-=sum(y*shadow_coeff(n,i,j) for i,y in ys.items())
        w.append(v)
    assert w[0]==0 and all(v>=0 for v in w),(n,w)
    if n in (16,20):
        assert circulant_purity(n,SHARP_WITNESS[n])==L+c,n
    else:
        m=n-1;rows=witness_base(n)
        assert len(rows)==m
        weight=graph_weight(rows)
        p=sum(parity_obj(m,j)*weight[j] for j in range(m+1))
        assert p==L+c,(n,p,L+c)
    print('ST-%d PASS: n=%d pi >= %s + (%s)*tau; positive A coefficients %s; %s'%(
       [4,8,10,14,16,20].index(n)+1,n,L,c,[j for j,v in enumerate(w) if v],
       'sharp coefficient witness'))
if __name__=='__main__':
    for n,data in CASES.items():check(n,*data)
