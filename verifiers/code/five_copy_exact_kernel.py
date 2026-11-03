"""Exact Q(sqrt(555)) rank and sparse left-kernel extraction for small blocks."""
import math

import flint
import numpy as np
import sympy as sp

from five_copy_young51_model import P


def exact_rank_and_kernel(A,B,r):
    """Verify field rank r and all-column annihilation; return compact kernel."""
    d=555;A0=15*A;n,m=A.shape
    assert A.shape==B.shape and 0<r<n
    assert max(int(np.abs(A0).max()),d*int(np.abs(B).max()))<2**63
    R=np.block([[A0,d*B],[B,A0]])
    assert flint.fmpz_mat([[int(v) for v in row] for row in R]).rank()==2*r
    root=int(sp.sqrt_mod(d,P));root=min(root,P-root)
    assert root*root%P==d%P
    X=(A0%P+(root*(B%P))%P)%P
    F=flint.nmod_mat(n,m,[int(v) for v in X.ravel()],P)
    Rr,rr=F.rref();assert rr==r
    J=[next(j for j in range(m) if int(Rr[i,j])) for i in range(r)]
    Y=X[:,J].T.copy()
    Ri,ri=flint.nmod_mat(r,n,[int(v) for v in Y.ravel()],P).rref();assert ri==r
    I=[next(j for j in range(n) if int(Ri[i,j])) for i in range(r)]
    D=np.block([[A0[np.ix_(I,J)],d*B[np.ix_(I,J)]],
                [B[np.ix_(I,J)],A0[np.ix_(I,J)]]])
    inv=flint.fmpq_mat(flint.fmpz_mat(D.tolist())).inv()
    assert all(inv[i,j]==inv[i+r,j+r] and inv[i,j+r]==d*inv[i+r,j]
               for i in range(r) for j in range(r))
    P0=[[inv[i,j] for j in range(r)] for i in range(r)]
    P1=[[inv[i+r,j] for j in range(r)] for i in range(r)]
    other=[i for i in range(n) if i not in I]
    numer0=[];numer1=[];denoms=[]
    for row in other:
        ar=[int(v) for v in A0[row,J]];br=[int(v) for v in B[row,J]]
        c0=[sum(ar[k]*P0[k][j]+d*br[k]*P1[k][j] for k in range(r)) for j in range(r)]
        c1=[sum(ar[k]*P1[k][j]+br[k]*P0[k][j] for k in range(r)) for j in range(r)]
        den=math.lcm(*[int(v.denom()) for v in c0+c1]);denoms.append(den)
        numer0.append([int(den*v) for v in c0]);numer1.append([int(den*v) for v in c1])
    fm=lambda Q:flint.fmpz_mat([[int(v) for v in row] for row in Q])
    Z=fm([[denoms[i] if i==j else 0 for j in range(len(other))] for i in range(len(other))])
    Q0,Q1=fm(numer0),fm(numer1)
    Ai,Bi=fm(A0[I]),fm(B[I]);Ao,Bo=fm(A0[other]),fm(B[other])
    assert Z*Ao==Q0*Ai+d*Q1*Bi
    assert Z*Bo==Q0*Bi+Q1*Ai
    assert all(v == 0 for row in numer1 for v in row), 'kernel is not rational'
    return {'field_d':d,'pivot_rows':I,'pivot_columns':J,'other_rows':other,
            'denominators':[str(x) for x in denoms],
            'coeff_rational':[list(map(str,row)) for row in numer0],
            'coeff_sqrt_d':[list(map(str,row)) for row in numer1]}
