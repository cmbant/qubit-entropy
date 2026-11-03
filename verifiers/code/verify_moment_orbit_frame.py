#!/usr/bin/env python3
"""Exact icosahedral Kramers-frame data for moment orbit rigidity (MO-2)."""
from fractions import Fraction as F
from itertools import combinations

N=7
u=[0]*128; v=[0]*128
pent={(1,3),(2,3),(2,4),(4,5),(1,5)}
for pair in combinations(range(6),2):
    bits=[0]*7
    for k in pair: bits[1+k]=1
    x=sum(bit<<(6-i) for i,bit in enumerate(bits))
    u[x]=-1 if pair in pent else 1
    u[x^127]=-u[x]
v[63]=v[64]=1
assert sum(t*t for t in u)==30 and sum(t*t for t in v)==2

def j(w):
    return [(-1)**x.bit_count()*w[x^127] for x in range(128)]

def pauli(w,site,axis):
    bit=1<<(6-site)
    if axis=='Z': return [(-1 if x&bit else 1)*w[x] for x in range(128)]
    if axis=='X': return [w[x^bit] for x in range(128)]
    # Y has purely imaginary entries; return its real coefficient up to -i.
    return [(-1 if x&bit else 1)*w[x^bit] for x in range(128)]

def frame_h_numerators(vv):
    ju,jv=j(u),j(vv)
    out=[]
    for i in range(7):
        row=[]
        for axis in 'XYZ':
            Au=pauli(ju,i,axis);Av=pauli(jv,i,axis)
            A=15*sum(a*b for a,b in zip(Au,u))+37*sum(a*b for a,b in zip(Av,vv))
            B=sum(a*b+c*d for a,b,c,d in zip(Au,vv,Av,u))
            row.append((A,B))
        out.append(row)
    return out

ico_h=frame_h_numerators(v)
for i in range(7):
    assert ico_h[i]==[(0,0),(0,0),(-376 if i==0 else -224,0)]
print('Direct spin-flip/Pauli contraction: h_s=-94/131, h_o=-56/131; transverse h=0; PASS')

# Independent negative control: an isolated-support amplitude phase flip
# preserves all one- and two-body marginals, but changes the frame shadow.
vflip=v.copy();vflip[63]=-1
support=[x for x in range(128) if u[x] or v[x]]
assert len(support)==32
assert all((63^x).bit_count()>=3 for x in support if x!=63)
flip_h=frame_h_numerators(vflip)
assert flip_h[0]==[(0,0),(0,0),(-524,0)]
assert all(row==[(0,0),(0,0),(-76,0)] for row in flip_h[1:])
assert F(1)+6*F(19,131)**2 != F(27652,17161)
print('Phase-flip negative regression: same pair marginals, different S1; PASS')

a,b=F(19,131),F(11,131)
hs,ho=-F(94,131),-F(56,131)
qs,qo=hs*hs,ho*ho
ds,do=6*a*a,a*a+5*b*b
S1=qs+6*qo
Q=qs*qs+6*qo*qo
R=ds*qs+6*do*qo
L=6*a*hs*ho+15*b*ho*ho
assert (qs,qo)==(F(8836,17161),F(3136,17161))
assert (ds,do)==(F(2166,17161),F(966,17161))
assert S1==F(27652,17161)
assert (R-ds*S1)/(do-ds)==6*qo
assert Q==qs*qs+6*qo*qo
print('S1=',S1,'Q=',Q,'R=',R,'L=',L)

# G has I_3 site blocks and T_ij off-site blocks. In its axial
# 7-dimensional block, multiply the proposed coefficients explicitly.
c=[-F(1,2)]+[-F(1,4)]*6
h=[hs]+[ho]*6
for i in range(7):
    gh=c[i]+sum((a if i==0 or j==0 else b)*c[j]
                for j in range(7) if j!=i)
    assert gh==h[i]
assert sum(c[i]*h[i] for i in range(7))==1
print('Gc=h and c^T h=1 exactly; projected residual vanishes; PASS')
