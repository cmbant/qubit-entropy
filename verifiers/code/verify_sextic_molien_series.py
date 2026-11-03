#!/usr/bin/env python3
"""HI-1--HI-4: exact Molien generating functions for sextic qubit invariants.

Site-symmetric LU invariants of bidegree (3,3) are invariants in
Sym^n(C + End([21])) under S3 x S3. Conjugation interchanges
left and right replica actions. This checker derives the untwisted
and twisted class determinants and proves H_-(t)=t^6 H_+(t).
All calculations are exact sympy integer/rational polynomials.
"""
import sympy as s

t=s.symbols("t")
I=s.eye(2)
# Genuine standard S3 representation from the sum-zero subspace.
T=s.Matrix([[-1,1],[0,1]])        # transposition (01)
C3=s.Matrix([[0,-1],[1,-1]])      # 3-cycle (012)
assert T*T==I and C3**3==I and T*C3*T==C3**-1
REP={"e":I,"s":T,"c":C3}

# Interchange two standard-representation factors: transpose of 2x2.
SWAP=s.zeros(5);SWAP[0,0]=1
for i in range(2):
    for j in range(2):
        SWAP[1+2*i+j,1+2*j+i]=1
assert SWAP**2==s.eye(5)

def local(a,b):
    return s.diag(s.ones(1),s.kronecker_product(REP[a],REP[b]))

def characteristic_denominator(a,b,twisted=False):
    M=(SWAP if twisted else s.eye(5))*local(a,b)
    x=s.symbols("x")
    return s.factor(t**5*M.charpoly(x).as_expr().subs(x,1/t))

for a,b,d in [
    ("e","e",(1-t)**5),
    ("e","s",(1-t)**3*(1+t)**2),
    ("s","s",(1-t)**3*(1+t)**2),
    ("e","c",(1-t)*(1+t+t*t)**2),
    ("s","c",(1-t)*(1+t*t+t**4)),
    ("c","c",(1-t)**3*(1+t+t*t))
]:
    assert s.factor(characteristic_denominator(a,b)-d)==0,(a,b)
print("HI-1 PASS: six untwisted S3xS3 class determinants")

for a,b,d in [
    ("e","e",(1-t)**4*(1+t)),
    ("e","s",(1-t)*(1-t**4)),
    ("e","c",(1-t**3)*(1-t**2)),
    ("s","s",(1-t)**4*(1+t)),
    ("s","c",(1-t)*(1-t**4)),
    ("c","c",(1-t**3)*(1-t**2))
]:
    assert s.factor(characteristic_denominator(a,b,True)-d)==0,(a,b)
print("HI-2 PASS: conjugation-twisted determinants depend only on product class")

H=s.Rational(1,36)*(
    1/(1-t)**5
    +15/((1-t)**3*(1+t)**2)
    +4/((1-t)*(1+t+t*t)**2)
    +12/((1-t)*(1+t*t+t**4))
    +4/((1-t)**3*(1+t+t*t))
)
Htw=s.Rational(1,6)*(
    1/((1-t)**4*(1+t))
    +3/((1-t)*(1-t**4))
    +2/((1-t**3)*(1-t**2))
)
even=s.factor((H+Htw)/2)
odd=s.factor((H-Htw)/2)
expected_even=(t**4-t**3+t**2-t+1)/(
    (1-t)**5*(1+t)**2*(1+t*t)*(1-t+t*t)*(1+t+t*t)**2)
assert s.factor(even-expected_even)==0
assert s.factor(odd-t**6*even)==0
print("HI-3 PASS: exact rational Hilbert series and H_odd=t^6 H_even")

def rational_taylor(f,N):
    num,den=s.cancel(f).as_numer_denom()
    p=s.Poly(num,t);q=s.Poly(den,t)
    out=[];q0=q.nth(0)
    for k in range(N):
        v=p.nth(k)-sum(q.nth(j)*out[k-j]
                       for j in range(1,min(k,q.degree())+1))
        out.append(s.cancel(v/q0))
    return out

all_dims=[1,1,2,3,5,7,12,15,22,29,39,50,67]
even_dims=[1,1,2,3,5,7,11,14,20,26,34,43,56]
odd_dims=[0,0,0,0,0,0,1,1,2,3,5,7,11]
for name,poly,target in [
    ("all",H,all_dims),("even",even,even_dims),("odd",odd,odd_dims)
]:
    assert rational_taylor(poly,len(target))==target,name
print("HI-4 PASS: exact sextic invariant counts through n=12 and no odd n<=5")
