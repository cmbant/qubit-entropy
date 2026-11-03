#!/usr/bin/env python3
"""SEH-1--SEH-3: rational interval and exact symmetry-adapted integer block
certificate for strict local maximality of mean 3|4 entropy at the seven-qubit
maximiser. No floating-point eigensolver, matrix positivity test or logarithm
sign is used in the certificate: both 128x128 blocks decouple canonically over Z
into 16 isotypic blocks of sizes 1x1, 2x2, and 4x4 certified by exact integer
Sylvester leading principal minors.
This certifies the local statement; the global bound is thm:seven-entropy-global-maximum (E7-1).
"""
import itertools
import math
from fractions import Fraction as F
import flint
import numpy as np
from verify_seven_entropy_stationarity import U,V,CUTS,N

D=10**8
A=16934604
B=26425139
SCALE=10**10
LLOG=10**14

def nearest_rat(num,den,scale=1):
    if den<0:num,den=-num,-den
    return (2*num*scale+den)//(2*den)

def arctanh_enclosure(t,terms=45):
    t=F(t)
    total=F(0);power=t
    for k in range(terms):
        total+=2*power/F(2*k+1)
        power*=t*t
    tail=2*abs(power)/(F(2*terms+1)*(1-t*t))
    return total-tail,total+tail

def ln_rational_enclosure(x):
    # log(x) = log(8*x)-3log2, exact power series tail enclosures.
    t=(8*x-1)/(8*x+1)
    lo,hi=arctanh_enclosure(t)
    l2,h2=arctanh_enclosure(F(1,3))
    return lo-3*h2,hi-3*l2

ln2lo,ln2hi=arctanh_enclosure(F(1,3))
cNUM=round(1/math.log(2)*LLOG)
assert max(abs(F(cNUM,LLOG)-1/ln2hi),
           abs(F(cNUM,LLOG)-1/ln2lo))<F(1,10**13)
logdata={}
for L in (4*A*A,3*A*A+B*B,8*A*A,A*A+B*B):
    if L in logdata:continue
    lo,hi=ln_rational_enclosure(F(L,D*D))
    num=round(math.log(L/(D*D))*LLOG)
    assert max(abs(F(num,LLOG)-lo),
               abs(F(num,LLOG)-hi))<F(1,10**13)
    logdata[L]=num

def rounded(arr,nom,den):
    """Exact integer nearest-grid enclosure: error <= .5/SCALE."""
    if den<0:nom,den=-nom,-den
    return (2*arr*nom*SCALE+den)//(2*den)

def projectors(M,contains_s):
    r=M@M.T
    es=[8*A*A,4*A*A,A*A+B*B] if contains_s else [4*A*A,3*A*A+B*B]
    out=[]
    for l in es:
        num=np.eye(8,dtype=object);den=1
        for h in es:
            if h==l:continue
            num=num@(r-h*np.eye(8,dtype=object))
            den*=l-h
        if den<0:num,den=-num,-den
        assert np.array_equal(r@num,l*num)
        assert np.array_equal(num@num,den*num)
        out.append((num,den))
    common=math.prod(d for _,d in out)
    su=sum((p*(common//d) for p,d in out),
           start=np.zeros((8,8),dtype=object))
    assert np.array_equal(su,common*np.eye(8,dtype=object))
    return es,out

def build_matrices():
    mn=(A*U+B*V).astype(object)
    Hreal=np.zeros((128,128),dtype=object)
    Himag=np.zeros((128,128),dtype=object)
    for S,T in CUTS:
        perm=S+T
        idx=np.arange(128).reshape((2,)*N).transpose(perm).reshape(128)
        M=mn.transpose(perm).reshape(8,16)
        es,ps=projectors(M,0 in S)
        proj=[(p@M,M.T@p@M) for p,d in ps]
        HR=np.zeros((8,16,8,16),dtype=object)
        HI=np.zeros((8,16,8,16),dtype=object)
        logR=np.eye(8,dtype=object)*SCALE
        for eig,(pn,den) in zip(es,ps):
            logR+=rounded(pn,logdata[eig],den*LLOG)
        for c in range(16):
            HR[:,c,:,c]+=logR
            HI[:,c,:,c]+=logR
        for l,(Pl,dl) in enumerate(ps):
            Al,Cl=proj[l]
            for m,(Pm,dm) in enumerate(ps):
                Am,Cm=proj[m]
                K2=(Pl[:,None,:,None]*Cm[None,:,None,:]
                    +Pm[:,None,:,None]*Cl.T[None,:,None,:])
                K1=(Al[:,None,None,:]*Am.T[None,:,:,None]
                    +Am[:,None,None,:]*Al.T[None,:,:,None])
                if l==m:
                    nom,den=1,2*es[l]*dl*dm
                else:
                    nom=logdata[es[l]]-logdata[es[m]]
                    den=2*LLOG*(es[l]-es[m])*dl*dm
                HR+=rounded(K2+K1,nom,den)
                HI+=rounded(K2-K1,nom,den)
        # Real/imaginary Hessian of the unnormalised entropy.
        HR=rounded(HR,-2*cNUM,35*LLOG*SCALE).reshape(128,128)
        HI=rounded(HI,-2*cNUM,35*LLOG*SCALE).reshape(128,128)
        Hreal[np.ix_(idx,idx)]+=HR
        Himag[np.ix_(idx,idx)]+=HI
    # Radial Lagrange multiplier; all values below are integer numerators.
    moment=(195*4*A*A*logdata[4*A*A]
          +40*(3*A*A+B*B)*logdata[3*A*A+B*B]
          +15*8*A*A*logdata[8*A*A]
          +30*(A*A+B*B)*logdata[A*A+B*B])
    nrm=30*A*A+2*B*B
    lam=nearest_rat(-cNUM*(moment+35*nrm*LLOG),
                    35*D*D*LLOG*LLOG,scale=SCALE)
    for H in (Hreal,Himag):
        for j in range(128):H[j,j]-=2*lam
    # The orbit generators are i*z and i*sigma_a^(j)z. Real Y,
    # imaginary X,Z,phase. Sign of Y convention does not affect GG^T.
    X=np.array([[0,1],[1,0]],dtype=object)
    Y=np.array([[0,-1],[1,0]],dtype=object)
    Z=np.diag([1,-1]).astype(object)
    gr=[];gi=[]
    for site in range(7):
        for typ,P in [('X',X),('Y',Y),('Z',Z)]:
            g=np.tensordot(P,mn,axes=(1,site))
            g=np.moveaxis(g,0,site).reshape(128)
            (gr if typ=='Y' else gi).append(g)
    gi.append(mn.reshape(128))
    Gr=np.stack(gr,axis=1)
    Gi=np.stack(gi,axis=1)
    # Exact rational Gram diagonally dominant, also proving orbit rank.
    for G in (Gr,Gi):
        low,_=gershgorin(G.T@G)
        assert F(low,D*D)>F(1,10)
    return tuple(-H+rounded(G@G.T,4,D*D)
                 for H,G in [(Hreal,Gr),(Himag,Gi)])

def gershgorin(M):
    lower=None;upper=None
    for j in range(M.shape[0]):
        d=int(M[j,j])
        r=sum(abs(int(M[j,k])) for k in range(M.shape[1]) if k!=j)
        lo,hi=d-r,d+r
        lower=lo if lower is None else min(lower,lo)
        upper=hi if upper is None else max(upper,hi)
    return lower,upper

def root_and_domain():
    lo=F(698287976009,10**13)
    hi=F(698287976010,10**13)
    def poly(x):
        p,q=x.numerator,x.denominator
        return (2**32*(q-2*p)**15
               -3**8*(q+8*p)**8*(q+28*p)**7)
    assert poly(lo)>0 and poly(hi)<0
    delta=F(1,200000000)
    a0,b0=F(A,D),F(B,D)
    assert (a0-delta)**2<(1-2*hi)/30<(a0+delta)**2
    assert (b0-delta)**2<lo<hi<(b0+delta)**2
    assert 32*delta**2 < F(3,10**8)**2
    amin,bmin=a0-delta,b0-delta
    for lam in (4*amin**2,3*amin**2+bmin**2,
                8*amin**2,amin**2+bmin**2):
        assert lam>F(1,12)
    for es in ([4*A*A,3*A*A+B*B],
               [8*A*A,4*A*A,A*A+B*B]):
        for el,em in itertools.combinations(es,2):
            assert abs(el-em)*64>D*D
    print('PASS SEH-1: root bracket and ||z*-z0||<3e-8; spectral floor 1/12')

def symmetry_blocks():
    pent={(1,3),(2,3),(2,4),(4,5),(1,5)}
    C=np.ones((6,6),dtype=np.int64)-np.eye(6,dtype=np.int64)
    for i,j in pent:C[i,j]=C[j,i]=-1
    pairs=list(itertools.combinations(range(6),2))
    L_disj=np.zeros((15,15),dtype=np.int64)
    for p_idx,(i,j) in enumerate(pairs):
        for q_idx,(k,l) in enumerate(pairs):
            if len({i,j,k,l})==4:
                u,v=[x for x in range(6) if x not in {i,j,k,l}]
                L_disj[p_idx,q_idx]=C[i,j]*C[k,l]*C[u,v]
    I15=np.eye(15,dtype=np.int64)
    P_L={
        3:(L_disj-2*I15)@(L_disj+I15)@(L_disj+3*I15)//2,
        -1:(L_disj-2*I15)@(L_disj-3*I15)@(L_disj+3*I15)//2,
        -3:-(L_disj-2*I15)@(L_disj-3*I15)@(L_disj+I15)//4,
    }
    def idx_set(s,S_sub):
        bits=[0]*7;bits[0]=s
        for i in S_sub:bits[i+1]=1
        return int(np.ravel_multi_index(bits,(2,)*7))
    E_supp_02=np.zeros((128,15),dtype=np.int64)
    E_supp_14=np.zeros((128,15),dtype=np.int64)
    E_flip_12=np.zeros((128,15),dtype=np.int64)
    E_flip_04=np.zeros((128,15),dtype=np.int64)
    for p_idx,(i,j) in enumerate(pairs):
        sg=int(C[i,j]);comp=[x for x in range(6) if x not in {i,j}]
        E_supp_02[idx_set(0,[i,j]),p_idx]=sg
        E_supp_14[idx_set(1,comp),p_idx]=-sg
        E_flip_12[idx_set(1,[i,j]),p_idx]=sg
        E_flip_04[idx_set(0,comp),p_idx]=sg
    e06=np.zeros(128,dtype=np.int64);e06[idx_set(0,range(6))]=1
    e10=np.zeros(128,dtype=np.int64);e10[idx_set(1,[])]=1
    e16=np.zeros(128,dtype=np.int64);e16[idx_set(1,range(6))]=1
    e00=np.zeros(128,dtype=np.int64);e00[0]=1
    blks=[([(E_supp_02+E_supp_14).sum(axis=1),e06+e10],[30,2]),
          ([(E_supp_02-E_supp_14).sum(axis=1),e06-e10],[30,2]),
          ([E_flip_12.sum(axis=1),E_flip_04.sum(axis=1),e16,e00],[15,15,1,1])]
    for lam in (3,-1,-3):
        P=P_L[lam]
        for sgn in (+1,-1):
            S=(E_supp_02+sgn*E_supp_14)@P
            d=int((S[:,np.flatnonzero((S**2).sum(axis=0))[0]]**2).sum())
            blks.append(([S],[d]))
        F1,F2=E_flip_12@P,E_flip_04@P
        d=int((F1[:,np.flatnonzero((F1**2).sum(axis=0))[0]]**2).sum())
        blks.append(([F1,F2],[d,d]))
    triples=list(itertools.combinations(range(6),3))
    E_in=np.zeros((20,6),dtype=np.int64);E_out=np.zeros((20,6),dtype=np.int64)
    B_03=np.zeros((128,20),dtype=np.int64);B_13c=np.zeros((128,20),dtype=np.int64)
    for t_idx,T in enumerate(triples):
        Tc=[x for x in range(6) if x not in T]
        B_03[idx_set(0,T),t_idx]=1;B_13c[idx_set(1,Tc),t_idx]=1
        for i in T:
            k,l=[x for x in T if x!=i];E_in[t_idx,i]=C[k,l]
        for i in Tc:
            k,l=[x for x in Tc if x!=i];E_out[t_idx,i]=C[k,l]
    F1_20=E_in;F2_20=E_out@C-E_in
    P8_40=40*np.eye(20,dtype=np.int64)-4*(F1_20@F1_20.T)-(F2_20@F2_20.T)
    for sgn in (+1,-1):
        M8=(B_03+sgn*B_13c)@P8_40
        blks.append(([M8],[int((M8[:,0]**2).sum())]))
    B_01=np.zeros((128,6),dtype=np.int64);B_11=np.zeros((128,6),dtype=np.int64)
    B_05=np.zeros((128,6),dtype=np.int64);B_15=np.zeros((128,6),dtype=np.int64)
    for i in range(6):
        comp_i=[x for x in range(6) if x!=i]
        B_01[idx_set(0,[i]),i]=1;B_11[idx_set(1,[i]),i]=1
        B_05[idx_set(0,comp_i),i]=1;B_15[idx_set(1,comp_i),i]=1
    for sgn in (-1,+1):
        blks.append(([(B_01+sgn*B_15)@C,B_11-sgn*B_05,(B_03+sgn*B_13c)@F1_20,(B_03+sgn*B_13c)@F2_20],[10,2,20,80]))
    return blks

def check_pd_int(M):
    k=M.shape[0]
    m=[[int(M[r,c]) for c in range(k)] for r in range(k)]
    prev=1
    for step in range(k):
        piv=m[step][step]
        if piv<=0:return False
        for r in range(step+1,k):
            for c in range(step+1,k):
                val=piv*m[r][c]-m[r][step]*m[step][c]
                assert val%prev==0
                m[r][c]=val//prev
        prev=piv
    return True

def verify_block_pd(basis,D_diag,B,shift):
    k=len(basis);L=int(np.lcm.reduce(D_diag))
    Q=np.zeros((k,k),dtype=object);M_scaled=np.zeros((k,k),dtype=object)
    for r in range(k):
        v_r=basis[r] if basis[r].ndim==1 else basis[r][:,np.flatnonzero((basis[r]**2).sum(axis=0))[0]]
        assert int(v_r@v_r)==D_diag[r]
        for c in range(k):
            v_c=basis[c] if basis[c].ndim==1 else basis[c][:,np.flatnonzero((basis[c]**2).sum(axis=0))[0]]
            val=int(v_r@(B@v_c))
            Q[r,c]=val-(shift*D_diag[r] if r==c else 0)
            M_scaled[r,c]=val*(L//D_diag[r])
    for c in range(k):
        assert np.array_equal(L*(B@basis[c]),sum(M_scaled[r,c]*basis[r] for r in range(k)))
    assert np.array_equal(Q,Q.T) and check_pd_int(Q)

def check_blocks_complete(blks):
    """Exact (Python-int / FLINT) checks that make the block test a statement about all of R^128:
    (i) within each block the basis members are mutually orthogonal, column by column
        (basis[r]^T basis[s] = 0 for r != s), so the diagonal Gram diag(D) used in
        verify_block_pd is the true Gram matrix of the vectors it tests;
    (ii) the columns of all 16 blocks together span R^128 (exact integer rank 128).
    The same 16 bases serve the real and the imaginary 128x128 block. By (ii) the product of the
    16 block characteristic polynomials annihilates Bhat, so every eigenvalue of Bhat is an
    eigenvalue of some block and the per-block bound is a bound on the whole sector."""
    cols=[]
    for basis,D_diag in blks:
        mats=[np.asarray(b).reshape(128,-1).astype(object) for b in basis]
        for r in range(len(mats)):
            for c in range(len(mats)):
                if r!=c:assert not np.any(mats[r].T@mats[c])
        cols+=[m[:,j] for m in mats for j in range(m.shape[1])]
    rank=flint.fmpz_mat([[int(x) for x in row] for row in np.column_stack(cols)]).rank()
    assert rank==128
    print('PASS SEH-2: block bases mutually orthogonal within each block; %d block columns span R^128 (exact rank %d)'
          %(len(cols),rank))

def certify():
    root_and_domain()
    shift=2393950000
    blks=symmetry_blocks()
    assert len(blks)==16
    check_blocks_complete(blks)
    for label,Bhat in enumerate(build_matrices()):
        for basis,D_diag in blks:
            verify_block_pd(basis,D_diag,Bhat,shift)
        bound=F(shift,SCALE)
        # ||B_exact-Bhat||_op < 10^4*3e-8+1e-5=0.00031.
        assert bound-F(31,100000)>F(239,1000)
        print('PASS SEH-2: block',label,'all 16 symmetry-adapted integer blocks (<=4x4) invariant and PD above',
              format(float(bound),'.6f'))
    print('PASS SEH-3: transverse Hessian < -0.239; strict LOCAL entropy maximum')
    print('Scope: strict local maximum; the global bound is thm:seven-entropy-global-maximum (E7-1).')

if __name__=='__main__':
    certify()
