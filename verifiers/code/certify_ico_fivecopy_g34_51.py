"""Exact icosahedral rank/kernel of the r=5 two-flip [5,1] block (candidate).

Portable if placed beside five_copy_young51_model.py and
verify_five_copy_site_algebra.py.  The 18 physical contractions use float64
only as an exact integer engine: an absolute-sum bound is below 2^53 for
every intermediate and every BLAS partial sum.  This computes the true
psi^3 (J_7 psi)^2 moment, not the unflipped five-copy moment.

The optional --output-dir writes compact exact A/B matrices and 125 sparse
kernel vectors; the default run verifies without creating files.
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import flint
import numpy as np
import opt_einsum as oe
import sympy as sp

import verify_five_copy_site_algebra as s5
from five_copy_young51_model import invariant_basis

P = 2147483629
PATTERNS = ((1,2,2,2,2,2,2), (2,1,2,2,2,2,2), (2,2,1,2,2,2,2))
CYCLE = (0,1,4,5,3,6,2)


def ico_integer_states():
    u = np.zeros((2,)*7, dtype=np.int64)
    v = np.zeros_like(u)
    pent = {(1,3),(2,3),(2,4),(4,5),(1,5)}
    for pair in itertools.combinations(range(6),2):
        sign = -1 if pair in pent else 1
        x = [0]*7
        for k in pair: x[1+k] = 1
        u[tuple(x)] = sign
        x = [1]*7
        for k in pair: x[1+k] = 0
        u[tuple(x)] = -sign
    v[(0,1,1,1,1,1,1)] = v[(1,0,0,0,0,0,0)] = 1
    assert np.count_nonzero(u) == 30 and np.count_nonzero(v) == 2
    # Manifest order-five site symmetry: fixes special site and ordinary 0,
    # cycles the five pentagon sites.  Thus three placement reps cover all 7.
    assert np.array_equal(u,u.transpose(CYCLE))
    assert np.array_equal(v,v.transpose(CYCLE))
    return u,v


def spinflip(z):
    out = np.zeros_like(z)
    for x in itertools.product(range(2),repeat=7):
        out[tuple(1-a for a in x)] = (-1)**sum(1-a for a in x)*z[x]
    return out


def local_maps():
    # Exact qubit lowering operator, with no float-derived matrix input.
    J = sp.zeros(32)
    for x in range(32):
        for c in range(5):
            mask = 1 << (4-c)
            if not (x & mask): J[x|mask,x] += 1
    maps = {}
    for m in (1,2):
        V = s5.V32[m]
        d, mult = V.cols, 6-2*m
        desc = []
        X = V
        for _ in range(mult):
            desc.append(X)
            X = J*X
        S = sp.Matrix.hstack(*[desc[k][:,i] for i in range(d) for k in range(mult)])
        L = (S.T*S).inv()*S.T
        den = sp.ilcm(*[int(x.q) for x in L])
        M = np.array([[int(den*x) for x in row] for row in L.tolist()],dtype=np.int64)
        assert sp.Matrix(M)*S == den*sp.eye(d*mult)
        maps[m] = M
    assert (maps[1].shape,maps[2].shape)==((16,32),(10,32))
    assert (int(np.abs(maps[1]).max()),int(np.abs(maps[2]).max()))==(24,3)
    return maps


def physical_sector(z,jz,pattern,maps):
    operands = []
    for c in range(5):
        operands += [z if c<3 else jz,[7*c+s for s in range(7)]]
    for s,m in enumerate(pattern):
        F = maps[m].astype(float).reshape((maps[m].shape[0],)+(2,)*5).transpose((1,2,3,4,5,0))
        operands += [F,[7*c+s for c in range(5)]+[35+s]]
    output = [35+s for s in range(7)]
    # Each copy state has support <=32 and entries <=5.  Exactly one local
    # [4,1] map and six [3,2] maps enter each summand.
    bound = 32**5 * 5**5 * 24 * 3**6
    # Each five-site monomial class has at most 5!=120 contributing tuples;
    # the Young numerator 6*Y_j-sum_l Y_l adds at most twelve such classes.
    assert bound < 2**53 and 12*120*bound < 2**63
    path,_ = oe.contract_path(*operands,output,optimize=oe.RandomGreedy(max_repeats=128))
    T = oe.contract(*operands,output,optimize=path)
    assert np.array_equal(T,np.rint(T))
    return T.astype(np.int64)


def young51_numerator(T,pattern):
    # The six choices of distinguished site realize the S6 Specht [5,1]
    # factor.  The five independent differences span it; the GL factor is
    # ker(Sym^5 V tensor V -> Sym^6 V), as in five_copy_young51_model.py.
    # Each monomial class sums at most 5!=120 tensor entries.  The Young
    # numerator 6*Y_j-sum_l Y_l has at most twelve such signed classes.
    assert 12*120*int(np.abs(T).max()) < 2**63
    order = [pattern.index(1)]+[i for i in range(7) if i!=pattern.index(1)]
    T = np.transpose(T,order).reshape(4,4,*([5,2]*6))
    T = T.transpose([0]+[2+2*j for j in range(6)]+[1]+[3+2*j for j in range(6)])
    T = T.reshape(4,*([5]*6),256)
    mons = list(itertools.combinations_with_replacement(range(5),5))
    mid = {m:i for i,m in enumerate(mons)}
    lookup = [mid[tuple(sorted(q))] for q in itertools.product(range(5),repeat=5)]
    Ys = []
    for j in range(6):
        Pj = T.transpose([0]+[1+i for i in range(6) if i!=j]+[1+j,7]).reshape(4,3125,5,256)
        Y = np.zeros((4,126,5,256),dtype=np.int64)
        for k in range(3125): Y[:,lookup[k]] += Pj[:,k]
        Ys.append(Y.reshape(2520,256))
    total = sum(Ys)
    return np.hstack([6*Y-total for Y in Ys[:5]])


def check_pentagon_orbit(T,M):
    """Exact one-step site permutation: only right Young/multiplicity columns move.

    The five distinguished-site differences give the standard S6 module:
    Z_j=6Y_j-sum_l Y_l, with Z_5=-sum_{j<5} Z_j.  Symmetrizing the
    other five GL slots makes a physical site permutation relabel j only;
    the GL(5) polynomial and H-invariant row coordinates do not move.
    """
    old_pat=PATTERNS[2]; old_ord=[i for i in range(7) if old_pat[i]!=1]
    new_special=CYCLE.index(old_pat.index(1))
    new_pat=tuple(1 if i==new_special else 2 for i in range(7))
    new_ord=[i for i in range(7) if i!=new_special]
    pi=[old_ord.index(CYCLE[i]) for i in new_ord]
    assert sorted(pi)==list(range(6))
    old=M.reshape(2520,5,4,*([2]*6))
    total=np.sum(old,axis=1)
    expected=[]
    for j in range(5):
        block=old[:,pi[j]] if pi[j]<5 else -total
        expected.append(block.transpose([0,1]+[2+pi[q] for q in range(6)]).reshape(2520,256))
    got=young51_numerator(T.transpose(CYCLE),new_pat)
    assert np.array_equal(got,np.hstack(expected))
    print('exact D5 pentagon placement covariance PASS',flush=True)


def exact_projection(U,M):
    maxu,maxm = int(np.abs(U).max()),int(np.abs(M).max())
    assert U.shape[0]*maxu*maxm < 2**63
    chunk = max(1,min(U.shape[0],(2**52)//max(1,maxu*maxm)))
    X = np.zeros((137,1280),dtype=np.int64)
    for i in range(0,U.shape[0],chunk):
        j=min(U.shape[0],i+chunk)
        Y = U[i:j].T.astype(float) @ M[i:j].astype(float)
        assert np.array_equal(Y,np.rint(Y))
        X += Y.astype(np.int64)
    return X


def interpolate(X):
    V = sp.Matrix([[t**k for k in range(6)] for t in range(6)])
    inv=V.inv(); den=sp.ilcm(*[int(v.q) for v in inv]); assert den==120
    N=np.array([[int(den*v) for v in row] for row in inv.tolist()],dtype=np.int64)
    maxx=int(np.abs(X).max())
    assert max(sum(abs(int(a)) for a in row) for row in N)*maxx < 2**63
    C=[]
    for k in range(6):
        Y=sum(int(N[k,t])*X[t] for t in range(6))
        assert not np.any(Y%den)
        C.append(Y//den)
    cm=[int(np.abs(c).max()) for c in C]
    assert max(sum(t**k*cm[k] for k in range(6)) for t in range(6))<2**63
    for t in range(6): assert np.array_equal(sum(t**k*C[k] for k in range(6)),X[t])
    abound=225*cm[0]+555*cm[2]+1369*cm[4]
    bbound=225*cm[1]+555*cm[3]+1369*cm[5]
    assert max(abound,bbound,15*abound,555*bbound)<2**63
    A=225*C[0]+555*C[2]+1369*C[4]
    B=225*C[1]+555*C[3]+1369*C[5]
    # The physical state has amplitudes (sqrt(15)u+sqrt(37)v)/sqrt(524).
    # Multiplying its five-copy projected tensor by sqrt(15)*524^(5/2)
    # yields the Q(sqrt(555)) matrix 15*A + sqrt(555)*B.
    return A,B,cm


def exact_rank_and_kernel(A,B):
    d=555; A0=15*A
    R=np.block([[A0,d*B],[B,A0]])
    M=flint.fmpz_mat([[int(v) for v in row] for row in R])
    assert M.rank()==24                 # rank over Q(sqrt(555)) = 12
    p=P; root=int(sp.sqrt_mod(d,p)); root=min(root,p-root)
    assert root*root%p==d%p
    X=(A0%p + (root*(B%p))%p)%p
    F=flint.nmod_mat(137,3840,[int(v) for v in X.ravel()],p)
    Rr,r=F.rref(); assert r==12
    J=[next(j for j in range(3840) if int(Rr[i,j])) for i in range(r)]
    Y=X[:,J].T.copy()
    Ri,ri=flint.nmod_mat(12,137,[int(v) for v in Y.ravel()],p).rref()
    assert ri==12
    I=[next(j for j in range(137) if int(Ri[i,j])) for i in range(12)]
    D=np.block([[A0[np.ix_(I,J)],d*B[np.ix_(I,J)]],
                [B[np.ix_(I,J)],A0[np.ix_(I,J)]]])
    inv=flint.fmpq_mat(flint.fmpz_mat(D.tolist())).inv()
    assert all(inv[i,j]==inv[i+12,j+12] and inv[i,j+12]==d*inv[i+12,j]
               for i in range(12) for j in range(12))
    P0=[[inv[i,j] for j in range(12)] for i in range(12)]
    P1=[[inv[i+12,j] for j in range(12)] for i in range(12)]
    other=[i for i in range(137) if i not in I]
    numer0=[]; numer1=[]; denoms=[]
    for row in other:
        ar=[int(v) for v in A0[row,J]]; br=[int(v) for v in B[row,J]]
        c0=[sum(ar[k]*P0[k][j]+d*br[k]*P1[k][j] for k in range(12)) for j in range(12)]
        c1=[sum(ar[k]*P1[k][j]+br[k]*P0[k][j] for k in range(12)) for j in range(12)]
        den=math.lcm(*[int(v.denom()) for v in c0+c1])
        denoms.append(den)
        numer0.append([int(den*v) for v in c0])
        numer1.append([int(den*v) for v in c1])
    assert all(not any(row) for row in numer1)  # kernel is rational
    fm=lambda X: flint.fmpz_mat([[int(v) for v in row] for row in X])
    Z=fm([[denoms[i] if i==j else 0 for j in range(125)] for i in range(125)])
    Q0,Q1=fm(numer0),fm(numer1)
    Ai,Bi=fm(A0[I]),fm(B[I]); Ao,Bo=fm(A0[other]),fm(B[other])
    assert Z*Ao==Q0*Ai+d*Q1*Bi
    assert Z*Bo==Q0*Bi+Q1*Ai
    print('rational kernel coefficients PASS',flush=True)
    return {'field_d':d,'pivot_rows':I,'pivot_columns':J,'other_rows':other,
            'denominators':[str(x) for x in denoms],
            'coeff_rational':[list(map(str,row)) for row in numer0],
            'coeff_sqrt_d':[list(map(str,row)) for row in numer1]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    u,v=ico_integer_states(); maps=local_maps(); U=invariant_basis()
    print('exact local maps and Young H-invariants: 42 site coordinates, 137 block coordinates',flush=True)
    xs=[]
    for t in range(6):
        z=u+t*v; jz=spinflip(z)
        assert np.array_equal(jz,jz.transpose((0,1,4,5,3,6,2)))
        blocks=[]
        for pattern in PATTERNS:
            T=physical_sector(z,jz,pattern,maps)
            M=young51_numerator(T,pattern)
            if t==0 and pattern==PATTERNS[2]: check_pentagon_orbit(T,M)
            blocks.append(exact_projection(U,M))
        X=np.hstack(blocks); xs.append(X)
        print('t',t,'three placement reps complete; max entry',int(np.abs(X).max()),flush=True)
    A,B,cm=interpolate(np.stack(xs))
    print('integer coefficient maxima',cm,'target pair maxima',int(np.abs(A).max()),int(np.abs(B).max()),flush=True)
    kernel=exact_rank_and_kernel(A,B)
    print('target exact rank 12 of 137; exact left kernel 125, full-column annihilation PASS',flush=True)
    data_dir=Path(__file__).resolve().parent.parent/'data'
    pair_path=data_dir/'five_copy_ico_g34_pair51.npz'
    kernel_path=data_dir/'five_copy_ico_g34_kernel51.json'
    assert pair_path.exists()==kernel_path.exists(), 'partial committed kernel artifacts'
    if pair_path.exists():
        saved=np.load(pair_path)
        assert np.array_equal(saved['A'],A) and np.array_equal(saved['B'],B)
        assert json.loads(kernel_path.read_text())==kernel
        print('committed compact artifacts reproduced exactly',flush=True)
    if args.output_dir:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output_dir/'five_copy_ico_g34_pair51.npz',A=A,B=B)
        with open(args.output_dir/'five_copy_ico_g34_kernel51.json','w') as f: json.dump(kernel,f)
    print('PASS',flush=True)


if __name__=='__main__': main()
