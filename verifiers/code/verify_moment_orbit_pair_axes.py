"""Exact Pauli data and constants for moment orbit rigidity (MO-1).

The icosahedral vector is (sqrt(15) u + sqrt(37) v)/sqrt(524).
Every Pauli expectation is computed as A/524+B*sqrt(555)/524,
with a separate Gaussian-imaginary component for odd Y counts.
"""
import itertools
from fractions import Fraction as F

N=7
u=[0]*128; v=[0]*128
pent={(1,3),(2,3),(2,4),(4,5),(1,5)}
for pair in itertools.combinations(range(6),2):
    sign=-1 if pair in pent else 1
    bits=[0]*7
    for k in pair: bits[1+k]=1
    x=sum(bit<<(6-i) for i,bit in enumerate(bits))
    u[x]=sign;u[x^127]=-sign
v[63]=v[64]=1
assert sum(x*x for x in u)==30 and sum(x*x for x in v)==2


def raw_expect(word,vv=None):
    """Integer A,B in (A+B sqrt(555))/524, real or imaginary."""
    if vv is None:vv=v
    xmask=zmask=ny=0
    for site,axis in word:
        bit=1<<(6-site)
        if axis in 'XY':xmask|=bit
        if axis in 'YZ':zmask|=bit
        ny+=axis=='Y'
    aa=bb=0
    for x in range(128):
        y=x^xmask
        phase=-1 if (x&zmask).bit_count()%2 else 1
        aa+=phase*(15*u[y]*u[x]+37*vv[y]*vv[x])
        bb+=phase*(u[y]*vv[x]+vv[y]*u[x])
    if ny%2:
        assert aa==bb==0, ('imaginary Hermitian expectation',word,aa,bb)
        return (0,0)
    if ny%4==2:aa=-aa;bb=-bb
    return aa,bb


def integer_expect(w,word):
    """Exact expectation for a real integer state, divided by its norm."""
    xmask=zmask=ny=0
    for site,axis in word:
        bit=1<<(6-site)
        if axis in 'XY':xmask|=bit
        if axis in 'YZ':zmask|=bit
        ny+=axis=='Y'
    num=sum((-1 if (x&zmask).bit_count()%2 else 1)*w[x^xmask]*w[x]
            for x in range(128))
    if ny%2:
        assert num==0
        return 0
    if ny%4==2:num=-num
    norm=sum(x*x for x in w)
    assert num%norm==0
    return num//norm


def wedge_sum_pair_state(w):
    T={}
    for i,j in itertools.combinations(range(7),2):
        T[i,j]=[[integer_expect(w,((i,a),(j,b))) for b in 'XYZ']
                for a in 'XYZ']
    total=0
    for i in range(7):
        M=[[0]*18 for _ in range(3)];k=0
        for j in range(7):
            if j==i:continue
            Q=T[i,j] if i<j else [[T[j,i][b][a] for b in range(3)] for a in range(3)]
            for a in range(3):
                for b in range(3):M[a][3*k+b]=Q[a][b]
            k+=1
        total+=sum((M[a][c]*M[b][d]-M[a][d]*M[b][c])**2
                   for a,b in itertools.combinations(range(3),2)
                   for c,d in itertools.combinations(range(18),2))
    return total


def mean_triple_purity(vv):
    """Exact A+B sqrt(555) coefficients for the average 3-cut purity."""
    A0=B0=0
    for S in itertools.combinations(range(7),3):
        other=tuple(i for i in range(7) if i not in S)
        U=[[0]*16 for _ in range(8)];V=[[0]*16 for _ in range(8)]
        for x in range(128):
            i=sum(((x>>(6-s))&1)<<(2-k) for k,s in enumerate(S))
            j=sum(((x>>(6-s))&1)<<(3-k) for k,s in enumerate(other))
            U[i][j]=u[x];V[i][j]=vv[x]
        A=[[15*sum(U[i][k]*U[j][k] for k in range(16))
            +37*sum(V[i][k]*V[j][k] for k in range(16))
            for j in range(8)] for i in range(8)]
        B=[[sum(U[i][k]*V[j][k]+V[i][k]*U[j][k] for k in range(16))
            for j in range(8)] for i in range(8)]
        A0+=sum(A[i][j]**2+555*B[i][j]**2
                for i in range(8) for j in range(8))
        B0+=2*sum(A[i][j]*B[i][j]
                  for i in range(8) for j in range(8))
    return F(A0,35*524**2),F(B0,35*524**2)


def main():
    T={}
    for i in range(7):
        for a in 'XYZ':assert raw_expect(((i,a),))==(0,0)
    for i,j in itertools.combinations(range(7),2):
        M=[]
        for a in 'XYZ':
            row=[]
            for b in 'XYZ':
                A,B=raw_expect(((i,a),(j,b)))
                assert B==0
                expected=76 if i==0 else 44
                assert A==(expected if a==b=='Z' else 0),(i,j,a,b,A)
                row.append(A)
            M.append(row)
        T[i,j]=M
    assert len(T)==21
    assert sum((T[i,j][2][2]//4)**2 for i,j in T)==3981
    for i in range(7):
        M=[[0]*18 for _ in range(3)]
        k=0
        for j in range(7):
            if j==i:continue
            Q=T[i,j] if i<j else [[T[j,i][b][a] for b in range(3)] for a in range(3)]
            for a in range(3):
                for b in range(3):M[a][3*k+b]=Q[a][b]
            k+=1
        assert all(M[a][c]*M[b][d]-M[a][d]*M[b][c]==0
                   for a,b in itertools.combinations(range(3),2)
                   for c,d in itertools.combinations(range(18),2))
    print('A1=0; all pair T matrices are c_ij e_z e_z^T, exact',flush=True)
    print('c_special=76/524=19/131; c_ordinary=44/524=11/131',flush=True)
    print('A2=(6*19^2+15*11^2)/131^2=3981/17161',flush=True)
    print('all seven incident-pair wedge sums vanish exactly; PASS',flush=True)
    # Negative control: two Bell pairs times a three-qubit GHZ state has
    # A1=0, but the Bell-pair correlation matrices have rank three.
    w=[0]*128
    for a,b,c in itertools.product(range(2),repeat=3):
        bits=(a,a,b,b,c,c,c)
        w[sum(bit<<(6-i) for i,bit in enumerate(bits))]=1
    assert sum(x*x for x in w)==8
    assert all(integer_expect(w,((i,a),))==0
               for i in range(7) for a in 'XYZ')
    assert [integer_expect(w,((0,a),(1,a))) for a in 'XYZ']==[1,-1,1]
    assert [integer_expect(w,((2,a),(3,a))) for a in 'XYZ']==[1,-1,1]
    assert wedge_sum_pair_state(w)==12
    print('negative control A1=0, pair-axis wedge C=12>0; PASS',flush=True)

    # The tetrad SOS gives a second independent constraint. GHZ4 times
    # GHZ3 has A1=C=0 and only ZZ pair tensors, yet disconnected nonzero
    # pair correlations violate a tetrad.
    w=[0]*128
    for a,b in itertools.product(range(2),repeat=2):
        bits=(a,a,a,a,b,b,b)
        w[sum(bit<<(6-i) for i,bit in enumerate(bits))]=1
    assert all(integer_expect(w,((i,a),))==0
               for i in range(7) for a in 'XYZ')
    assert wedge_sum_pair_state(w)==0
    assert integer_expect(w,((0,'Z'),(1,'Z')))==1
    assert integer_expect(w,((4,'Z'),(5,'Z')))==1
    assert integer_expect(w,((0,'Z'),(4,'Z')))==0
    assert integer_expect(w,((1,'Z'),(5,'Z')))==0
    print('negative control A1=C=0, but disjoint-edge tetrad=1; PASS',flush=True)

    a,b=F(19,131),F(11,131)
    def c(i,j):return a if i==0 or j==0 else b
    E2=sum((c(i,j)**2 for i,j in itertools.combinations(range(7),2)),F(0))
    E3=sum((c(i,j)*c(j,k)*c(k,i)
            for i,j,k in itertools.combinations(range(7),3)),F(0))
    E4=F(0)
    for i,j,k,l in itertools.combinations(range(7),4):
        products=(c(i,j)*c(j,k)*c(k,l)*c(l,i),
                  c(i,j)*c(j,l)*c(l,k)*c(k,i),
                  c(i,k)*c(k,j)*c(j,l)*c(l,i))
        assert products[0]==products[1]==products[2]
        E4+=sum(products)/3
        assert c(i,j)*c(k,l)==c(i,k)*c(j,l)==c(i,l)*c(j,k)
    B1=sum((c(i,j)**4 for i,j in itertools.combinations(range(7),2)),F(0))
    assert E2==F(3981,17161)
    assert E3==F(86185,2248091)
    assert E4==F(1093235,294499921)
    assert B1==F(1001541,294499921)
    s=(E2**2+2*E4-B1)/(2*E3)
    p2=s*s-2*E2;p3=s**3-3*E2*s+3*E3
    S2=p2-s*s/7;S3=p3-3*s*p2/7+2*s**3/49
    assert s==F(1087,1441) and p2==F(218167,2076481)
    assert p3==F(57675247,2992209121)
    assert S2>0 and S3>0 and S3*S3==F(25,42)*S2**3
    print('E2,E3,E4,B1 exact; Newton recovers p1=1087/1441',flush=True)
    print('centered skewness saturates 5/sqrt(42); one high and six low weights',flush=True)
    print('pair-weight rigidity arithmetic PASS',flush=True)
    # One- and two-body marginals alone are not an orbit certificate.
    # Basis index 63 has Hamming distance at least three from every other
    # occupied ico string, so changing only its phase cannot affect them.
    support=[i for i in range(128) if u[i] or v[i]]
    assert len(support)==32
    assert all((63^j).bit_count()>=3 for j in support if j!=63)
    vv=v.copy();vv[63]=-1
    words=[((i,a),) for i in range(7) for a in 'XYZ']
    words += [((i,a),(j,b)) for i,j in itertools.combinations(range(7),2)
              for a,b in itertools.product('XYZ',repeat=2)]
    assert len(words)==210
    assert all(raw_expect(word,vv)==raw_expect(word,v) for word in words)
    assert mean_triple_purity(v)==(F(121,917),F(0))
    assert mean_triple_purity(vv)==(F(33367,240254),F(0))
    assert F(33367,240254)-F(121,917)==F(1665,240254)>0
    print('phase-flipped index 63: all 210 one/two-body Pauli values unchanged',flush=True)
    print('triple purity rises by exactly 1665/240254; distinct LU orbit PASS',flush=True)


if __name__=='__main__':main()
