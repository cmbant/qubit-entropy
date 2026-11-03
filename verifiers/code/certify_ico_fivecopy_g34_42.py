"""Exact icosahedral rank/kernel of the r=5 two-flip [4,2] block (candidate)."""
import argparse
import itertools
import json
import math
from pathlib import Path

import flint
import numpy as np
import sympy as sp

from certify_ico_fivecopy_g34_51 import (P, PATTERNS, ico_integer_states,
    CYCLE, local_maps, physical_sector, spinflip, interpolate)
from five_copy_young42_model import model_and_projector, invariant_basis
from five_copy_young51_model import monomials


def pair_grouping():
    m2,m4=[monomials(k) for k in (2,4)]
    i2,i4=[{e:j for j,e in enumerate(m)} for m in (m2,m4)]
    def idx(q, lookup):
        e=tuple(q.count(j) for j in range(5))
        return lookup[e]
    a=np.array([idx(q,i4) for q in itertools.product(range(5),repeat=4)],dtype=np.int32)
    b=np.array([idx(q,i2) for q in itertools.product(range(5),repeat=2)],dtype=np.int32)
    group=(a[:,None]*15+b[None,:]).ravel()
    sort=np.argsort(group,kind='stable')
    starts=np.r_[0,np.where(np.diff(group[sort]))[0]+1]
    assert len(starts)==1050
    return sort,starts


def young42_image(T,pattern,P40,sort,starts):
    order=[pattern.index(1)]+[i for i in range(7) if i!=pattern.index(1)]
    T=np.transpose(T,order).reshape(4,4,*([5,2]*6))
    T=T.transpose([0]+[2+2*j for j in range(6)]+[1]+[3+2*j for j in range(6)])
    T=T.reshape(4,*([5]*6),256)
    mats=[]
    for pair in itertools.combinations(range(6),2):
        rest=[j for j in range(6) if j not in pair]
        X=T.transpose([0]+[1+j for j in rest+list(pair)]+[7]).reshape(4,15625,256)
        maxx=int(np.abs(X).max())
        # One Sym^4 x Sym^2 monomial class has at most 4!*2!=48 tuples.
        assert 48*maxx < 2**63
        Y=np.add.reduceat(X[:,sort,:],starts,axis=1).reshape(4,1050,256)
        maxy=int(np.abs(Y).max())
        assert 96*maxy < 2**63
        Z=np.stack([P40@Y[i] for i in range(4)],axis=0)
        assert np.max(np.abs(Z)) < 2**53
        mats.append(Z.reshape(4200,256))
    return np.hstack(mats)


def check_pentagon_orbit(T,M,P40,sort,starts):
    """An exact site cycle permutes only 15 pair and 256 spin columns."""
    old_pat=PATTERNS[2]; old_ord=[i for i in range(7) if old_pat[i]!=1]
    new_special=CYCLE.index(old_pat.index(1))
    new_pat=tuple(1 if i==new_special else 2 for i in range(7))
    new_ord=[i for i in range(7) if i!=new_special]
    pi=[old_ord.index(CYCLE[i]) for i in new_ord]
    assert sorted(pi)==list(range(6))
    pairs=list(itertools.combinations(range(6),2)); pair_id={p:i for i,p in enumerate(pairs)}
    old=M.reshape(4200,15,4,*([2]*6))
    expected=[]
    for a,b in pairs:
        idx=pair_id[tuple(sorted((pi[a],pi[b])))]
        block=old[:,idx]
        expected.append(block.transpose([0,1]+[2+pi[q] for q in range(6)]).reshape(4200,256))
    got=young42_image(T.transpose(CYCLE),new_pat,P40,sort,starts)
    assert np.array_equal(got,np.hstack(expected))
    print('exact D5 pentagon placement covariance PASS',flush=True)


def project_exact(U,M):
    maxu,maxm=int(np.abs(U).max()),int(np.abs(M).max())
    assert U.shape[0]*maxu*maxm < 2**63
    chunk=max(1,min(U.shape[0],(2**52)//max(1,maxu*maxm)))
    X=np.zeros((148,3840),dtype=np.int64)
    for i in range(0,U.shape[0],chunk):
        j=min(U.shape[0],i+chunk)
        Y=U[i:j].T.astype(float)@M[i:j].astype(float)
        assert np.array_equal(Y,np.rint(Y))
        X+=Y.astype(np.int64)
    return X


def exact_rank_and_kernel(A,B):
    d=555; A0=15*A; n=A.shape[0]; m=A.shape[1]
    # A nonzero 20x20 field minor below proves rank >=20.  The 128
    # independent exact left-kernel identities checked below prove rank <=20.
    root=int(sp.sqrt_mod(d,P)); root=min(root,P-root)
    assert root*root%P==d%P
    X=(A0%P+(root*(B%P))%P)%P
    F=flint.nmod_mat(n,m,[int(v) for v in X.ravel()],P)
    Rr,r=F.rref(); assert r==20
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
    assert all(not any(row) for row in numer1)  # kernel is rational
    fm=lambda X:flint.fmpz_mat([[int(v) for v in row] for row in X])
    Z=fm([[denoms[i] if i==j else 0 for j in range(len(other))] for i in range(len(other))])
    Q0,Q1=fm(numer0),fm(numer1)
    Ai,Bi=fm(A0[I]),fm(B[I]);Ao,Bo=fm(A0[other]),fm(B[other])
    assert Z*Ao==Q0*Ai+d*Q1*Bi
    assert Z*Bo==Q0*Bi+Q1*Ai
    print('rational kernel coefficients PASS',flush=True)
    return {'field_d':d,'pivot_rows':I,'pivot_columns':J,'other_rows':other,
            'denominators':[str(x) for x in denoms],
            'coeff_rational':[list(map(str,row)) for row in numer0],
            'coeff_sqrt_d':[list(map(str,row)) for row in numer1]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    u,v=ico_integer_states();maps=local_maps()
    B0,P40,E=model_and_projector();U=invariant_basis(B0,E)
    sort,starts=pair_grouping()
    print('exact [4,2] model and 148 H-invariant coordinates',flush=True)
    xs=[]
    for t in range(6):
        z=u+t*v;jz=spinflip(z);blocks=[]
        for pat in PATTERNS:
            T=physical_sector(z,jz,pat,maps)
            M=young42_image(T,pat,P40,sort,starts)
            if t==0 and pat==PATTERNS[2]: check_pentagon_orbit(T,M,P40,sort,starts)
            blocks.append(project_exact(U,M))
        X=np.hstack(blocks);xs.append(X)
        print('t',t,'three placement reps complete; max entry',int(np.abs(X).max()),flush=True)
    A,B,cm=interpolate(np.stack(xs))
    print('integer coefficient maxima',cm,'target pair maxima',int(np.abs(A).max()),int(np.abs(B).max()),flush=True)
    kernel=exact_rank_and_kernel(A,B)
    print('target exact rank 20 of 148; exact left kernel 128, full-column annihilation PASS',flush=True)
    data_dir=Path(__file__).resolve().parent.parent/'data'
    pair_path=data_dir/'five_copy_ico_g34_pair42.npz'
    kernel_path=data_dir/'five_copy_ico_g34_kernel42.json'
    assert pair_path.exists()==kernel_path.exists(), 'partial committed kernel artifacts'
    if pair_path.exists():
        saved=np.load(pair_path)
        assert np.array_equal(saved['A'],A) and np.array_equal(saved['B'],B)
        assert json.loads(kernel_path.read_text())==kernel
        print('committed compact artifacts reproduced exactly',flush=True)
    if args.output_dir:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output_dir/'five_copy_ico_g34_pair42.npz',A=A,B=B)
        with open(args.output_dir/'five_copy_ico_g34_kernel42.json','w') as f:json.dump(kernel,f)
    print('PASS',flush=True)


if __name__=='__main__':main()
