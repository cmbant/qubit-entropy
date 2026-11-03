#!/usr/bin/env python3
"""SL: exact algebra for the analytic HT sliver replacement.

Proved in: sup_renyi.tex (lem:fqe-cluster-sliver and its proof). No continuum interval sign tests or subdivision.
Finite logarithm/power constants use integer-root and atanh enclosures.
Taylor, coordinate reductions, projection and Hermite remainders are analytic.
"""
from fractions import Fraction as F
from pathlib import Path
import sympy as s

def tex_source(name):
    """The TeX source, in ../paper/ or next to code/."""
    root = Path(__file__).resolve().parents[1]
    return next(p for p in (root/name, root.parent/'paper'/name) if p.exists())

R = s.Rational
QA = F(1587, 1000)
EPS, PH, P2, XM = F(1, 100), F(47, 100), F(29, 100), F(1, 25)
THETA_A = (F(35425, 10000), F(1872, 10000), F(11235, 10000))
u,v,e,k=s.symbols('u v e k')
U,V,E=R(1,16),R(1,20),R(1,100)
klo,khi=R(13856,10000),R(13857,10000)
def main():
    assert klo*klo < R(48,25) < khi*khi

    # SL-1: the only transcendental constants, at finitely many rational points.
    from rational_intervals import LOG2, SCALE
    Llo, Lhi = F(693147, 1000000), F(693148, 1000000)
    assert Llo < F(LOG2.lo,SCALE) < F(LOG2.hi,SCALE) < Lhi
    K, ca, fac = F(7801,2000), F(5,2), F(51,50)
    assert QA/((QA-1)*Llo) < K
    assert fac > ca*(QA-1)*Lhi
    assert 16**13*50**1000 > 51**1000
    assert (QA*QA/(2*(1-QA/16))-QA*(QA-1)/2)/((QA-1)*Llo) < F(12,5)
    # Head/tail coefficient bounds in (SH).
    assert F(16,15)**3 < F(28,25)**2
    assert (1+2*U)-(1+R(3,2)*U)*(1+R(7,25)*U) > 0
    assert F(159,100)*F(59,100)/2*F(16,15)**2 < F(27,50)
    assert QA/(2*Lhi) > F(113,100)
    assert F(27,50)**2/(2*(QA-1)*Llo) < F(3,8)
    assert 16**2 > 3**5 and 15**3 > 50**2
    assert QA/(2*Lhi)*3*F(199,200) > 3
    print('SL-1 PASS: binary/head/tail bounds and finite exact logarithm/power constants')

    rows={'S-':(R(2,125),R(7,500),-R(571,100),-R(33,10),-R(281,100),6,R(185,100),8),'S+':(R(2,125),R(7,500),-6,R(69,50),-R(103,20),0,R(147,100),16),'B-':(R(2,125),R(19,100),-6,3,-5,0,R(1,2),0),'B+':(R(2,125),R(19,100),-R(571,100),-R(3,2),-R(5,2),0,R(1,2),0)}
    def rational_coeff(expr):
     pol=s.Poly(expr,k)
     return sum(c*(klo if c>=0 else khi)**i[0] for i,c in pol.terms())
    for name,row in rows.items():
     ch,sg=name[0],-1 if name[1]=='-' else 1
     al,be,ga,K=R('3.5425'),R('.1872'),R('1.1235'),R('3.9005')
     p=R(1,2)-u+u*u+v/2;t=R(1,4)-3*u/4+3*u*u/4+R(3,4)*((1-u) if ch=='S' else u)*v
     delta=s.symbols('delta')
     lam=((1-u+delta)/2,(1-u-delta)/2,u/2,u/2) if ch=='S' else ((1-u)/2,(1-u)/2,(u+delta)/2,(u-delta)/2)
     assert s.expand(sum(lam)-1)==0
     assert s.expand(sum(z*z for z in lam)-p.subs(v,delta*delta))==0
     assert s.expand(sum(z**3 for z in lam)-t.subs(v,delta*delta))==0
     X=R(1,2)+e-p;fc=-1+8*e+9*p-12*t;ff=fc+sg*k*X
     hh=1+K*u+R(12,5)*u*u-(R(113,100)*v+R(3,2)*u*v-R(3,8)*v*v if ch=='S' else 3*v)
     G=s.expand(1+al/6+3*be/2-5*ga/36-hh-al*(p-R(1,3))-be*(17*p-12*t-4)+ga*(p*p-R(1,9))+al*e+be*ff-ga*(ff*ff+R(17,2)*e+2*e*e)/6-be/2+ga/24)
     a,b,c,d,f,g,A,B=map(R,row)
     target=a*u+b*v+c*u*u+d*u*v+f*v*v+g*u*u*v
     # residual constant in e: lower all negative higher terms by quadratics
     out={(1,0):R(0),(0,1):R(0),(2,0):R(0),(1,1):R(0),(0,2):R(0)}
     for (i,j),co in s.Poly(G.subs(e,0)-target,u,v).terms():
      co=rational_coeff(co)
      if (i,j) in out:out[(i,j)]+=co
      elif co<0:
       if i>=2:out[(2,0)]+=co*U**(i-2)*V**j
       elif i:out[(1,1)]+=co*V**(j-1)
       elif j>=2:out[(0,2)]+=co*V**(j-2)
       else:assert co==0
     assert min(out.values())>=0
     # e terms: preserve positive multiples of v and charge negative v terms to v.
     ee=s.cancel((G-G.subs(e,0))/e)
     c0,c1=R(0),R(0)
     for (i,j,l),co in s.Poly(ee,u,v,e).terms():
      co=rational_coeff(co)
      if (i,j,l)==(0,0,0):c0+=co
      elif (i,j,l)==(0,1,0) and ch=='S':c1+=co
      elif co<0:
       if ch=='S' and j:c1+=co*U**i*V**(j-1)*E**l
       else:c0+=co*U**i*V**j*E**l
     assert c0>=A and c1>=B

    print('SL-2 PASS: four exact polynomial lower bounds and epsilon coefficients')

    # SL-3: domain and projection constants.
    assert R(49,250)/(R(9,2)*R(1,2)) < R(9,100)
    assert R(1,10)-R(1,10)**2 > R(3,100)+R(9,100)/2
    assert R(49,250)/(R(9,2)*R(4,5)) < R(11,200)
    assert U-U*U > R(3,100)+R(11,200)/2
    assert R(49,250)/(R(9,2)*R(7,8)) < V
    assert R(1,2)-R(1,30)+R(3,2)*R(1,30)**2 < R(47,100)
    D=R(26,5)-9*u
    N=R(3,50)+R(7,5)*(u-u*u)
    Vfun=N/D
    assert R(47,5)/(R(26,5)-9*U) < R(21,10)
    for name,row in rows.items():
        a,b,c,d,f,g,A,B=map(R,row)
        if name[0]=='S':
            c0=max(R(0),-b+max(R(0),-d)*U)
            assert A-R(21,10)*c0+min(R(0),B+R(21,5)*f)*V > R(9,10)
        else:
            assert b+min(R(0),d)*U+f*U*U > R(3,40)
    print('SL-3 PASS: domain and epsilon removal, exact rational')

    # SL-4: the one-variable endpoint signs. Negative f makes P concave in v.
    def minorant(Z,z):
        upper,contact=Z**5,z**5
        return z**8+R(8,5)*z**3*(u-contact)+(Z**8-z**8-R(8,5)*z**3*(upper-contact))/(upper-contact)**2*(u-contact)**2
    assert U<R(23,40)**5
    assert 9*R(1,150)*(1-R(1,150))*(1-2*R(1,150)) < R(3,50)
    assert 9*R(1,140)*(1-R(1,140))*(1-2*R(1,140)) > R(3,50)
    assert R(51,1000)**5 < R(1,140)**3
    printed={
        'S-': [793,27467,39030,38325,27813,9606],
        'S+': [238,51693,86240,106860,116069],
    }
    x=s.symbols('x')
    for name,lower in printed.items():
        a,b,c,d,f,g,_,_=map(R,rows[name])
        ca=R(5,2)
        DD=2*b-c-2*min(d,R(139,140)*d)-4*f
        assert a+2*b>0 and a+2*b+ca*R(51,1000)-DD/140>0
        Z=R(23,40)
        P=a*u+b*Vfun+c*u*u+d*u*Vfun+f*Vfun*Vfun+g*u*u*Vfun+ca*minorant(Z,R(2,5))
        poly=s.Poly(s.cancel(P*D*D),u)
        shift=s.Poly(poly.as_expr().subs(u,R(1,150)+(Z**5-R(1,150))*x),x)
        n=shift.degree()
        assert len(lower)==n+1
        co=[sum(shift.nth(j)*s.binomial(i,j)/s.binomial(n,j) for j in range(i+1)) for i in range(n+1)]
        assert all(cc>R(ll,1000000)>0 for cc,ll in zip(co,lower))
    tex=tex_source('sup_renyi.tex').read_text()
    def table(tag):
        block=tex.split('% '+tag+'-BEGIN\n')[1].split('% '+tag+'-END')[0]
        return [line.strip().rstrip('\\').split('&') for line in block.strip().splitlines()]
    for row in table('SL-COEFFICIENTS'):
        assert tuple(map(R,row[1:])) == tuple(map(R,rows[row[0]]))
    for row in table('SL-BERNSTEIN'):
        numbers=[int(value) for value in row[2:] if value]
        assert numbers == printed[row[0]] and int(row[1])+1 == len(numbers)
    print('SL-4 PASS: endpoint signs and all 11 printed Bernstein coefficient bounds; no subdivision')


if __name__ == "__main__":
    main()
