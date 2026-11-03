#!/usr/bin/env python3
"""CG-1--CG-5: displayed algebra/arithmetic in the capped cubic proof.

Proved in: renyi_capped_cubic.tex and renyi_binary_quartic.tex. CG-1 checks the CG
column of tab:fqe-moment-rectangle (sup_renyi.tex) through
moment_rectangle.check_rectangle. The analytic
continuation, zero location and Cauchy integral are proved in the text.
This checks their kernel algebra and the two rational endpoint estimates;
there are no polynomial certificates, subdivisions or parameter coverings.
"""
from fractions import Fraction as F
import sympy as s
from rational_intervals import power, log_interval, LOG2, SCALE
import moment_rectangle as MR

R = s.Rational
q, r = R(1587,1000), R(587,1000)
LM, LP = R(693147,10**6), R(693148,10**6)
X, D = R(17,100), R(6,25)
A, B = 2*q*r, R(8,3)*q*r*(q-2)
C = R(8,3)*q*r*(q-2)*(q-3)
D5 = R(2,3)*(q-4)*C


def moments():
    assert LM < R(LOG2.lo,SCALE) < R(LOG2.hi,SCALE) < LP
    assert R(186,100)<A<R(187,100) and R(-103,100)<B<R(-102,100)
    assert R(144,100)<C<R(146,100) and R(-234,100)<D5<R(-233,100)
    assert R(17,300) < D**2
    # Coarse V,W,N bounds and the quadratic lower bound for V^2-2W.
    assert -R(102,100)+2*R(146,100)*D < 0
    assert R(186,100)-R(103,100)*D+R(144,100)*D**2 > R(5,3)
    assert R(187,100)+R(103,100)*D+R(146,100)*D**2 < R(9,4)
    assert R(144,100)/4-R(234,100)*D > R(-21,100)
    assert R(146,100)/4+R(234,100)*D < 1
    assert R(5,3)-X*R(21,100)>0
    assert 1+X*R(9,4)+X**2 < R(3,2)
    assert R(186,100)**2-R(146,100)/2 == R(1706,625)
    linear = 2*A*B-2*D5
    assert abs(linear)<R(9,10)
    assert R(102,100)**2+2*R(186,100)*R(144,100)-2*R(103,100)*R(146,100)*D>5
    assert R(1706,625)-R(81,2000)>R(8,3)
    assert (R(8,3)-2*R(9,4)*R(21,100)*X)/R(3,2)**2>1
    # m = 1 > 2 rho gamma, rho = 21/50 = (3/5)(7/10) > rL.
    assert 1 > 2*R(21,50)*R(11,10) and r*LP < R(21,50)
    assert -1+R(11,5)*r*LP<0
    slope = R(103,100)+2*R(146,100)*D+X*R(234,100)
    assert slope < R(213,100)
    assert (2*R(144,100)-X*slope**2)/(R(3,5)*R(7,10)*R(3,2)**2)>2
    assert (2*R(144,100)-X*R(213,100)**2)/(R(21,50)*R(3,2)**2)>2
    # The CG column of tab:fqe-moment-rectangle: (MR1) but Q, (MR2) and the
    # point values (MR3) at v_* = 0, z = XA+X^2C/4, by the shared exact check.
    col = MR.TABLE['CG']
    assert (col['X'], -col['vmin'], col['vmax'], col['vstar']) == (X, D, D, 0)
    assert (col['b'], col['c'], col['gamma']) == (R(8,3), R(171,40), R(11,10))
    assert (col['Nplus'], col['Vplus'], col['Wminus']) == (R(3,2), R(9,4), R(21,100))
    assert (col['Q'], col['m'], col['S'], col['kappa']) == (R(8,3), 1, R(213,100), 2)
    assert (col['f0'], col['eta'], col['k'], col['L']) == (R(1,250), R(1,20), 8, (LM, LP))
    assert tuple(R(e.numerator, e.denominator) for e in MR.coefficients()) == (A, B, C, D5)
    mu = MR.check_rectangle('CG')
    assert mu == R(1,250)-R(1,20)**2/4 > 0
    print('CG-1 moment rectangle (CG column, shared check): concavity in x, tangent square and the two rational point bounds PASS')


def binary_integral():
    v,t,a=s.symbols('v t a',positive=True)
    assert s.simplify(s.diff(v/(t*(t-v)),v,3)-6/(t-v)**4)==0
    assert s.simplify(s.diff(s.log(1+v/a),v,3)-2/(a+v)**3)==0
    assert s.expand(((1+s.sqrt(v))**2+(1-s.sqrt(v))**2)/2-(1+v))==0
    D=s.Function('D')(v)
    dd=(v-D)/(v*(1-v))
    ddd=((1-D)/(1-v)-s.diff(D,v))/(1-v)-(s.diff(D,v)-D/v)/v
    assert s.simplify(s.diff(dd,v)-ddd)==0
    # A positive third derivative makes the quartic correction monotone.
    # Only the two endpoint entropies use numerical constants, all rational.
    # -ln(1-z) = sum z^k/k <= z+z^2/2+z^3/(3(1-z)).
    cases=[(R(11,40),R(3,100),R(19,5000),R(1191,1250),R(4783,5000),R(1093,10000)),
           (R(3,8),R(1,12),R(193,10000),R(871,1000),R(8903,10000),R(5713,20000))]
    for a,u,lo1,lo2,m0,hcap in cases:
        assert u**1587>lo1**1000 and (1-u)**1587>lo2**1000
        assert lo1+lo2==m0
        z=1-m0
        minus_log=z+z**2/2+z**3/(3*(1-z))
        assert minus_log<r*LM*hcap
        d=u*(1-u)
        assert 4*d-a*d*(1-4*d)>hcap
    print('CG-2 positive integral kernel and divided-difference identities; both rational binary endpoints PASS')


def binary_consequences():
    assert R(11,40)==R(11,10)/4
    assert R(1,12)*(1-R(1,12))<R(1,8)
    assert R(3,8)*min(R(11,144)*R(25,36),R(44,225)*R(49,225))>R(187,12500)
    assert R(11,10)*R(2,25)*R(17,100)==R(187,12500)
    print('CG-3 binary quartics imply the required slack by monotonicity and two concave endpoint bounds PASS')


def pairs_and_assembly():
    u,v,z,t=s.symbols('u v z t',positive=True)
    p=R(1,2)-u+u*u+v/2
    ts=R(1,4)-R(3,4)*u+R(3,4)*u*u+R(3,4)*(1-u)*v
    tb=R(1,4)-R(3,4)*u+R(3,4)*u*u+R(3,4)*u*v
    for third,coefficient in ((ts,1+2*u),(tb,3-2*u)):
        assert s.expand(R(10,3)-6*p+R(8,3)*third-(1+4*u-4*u*u-coefficient*v))==0
    assert 1-3*R(4,15)+3*R(4,15)**2<R(21,50)
    assert R(1,2)-R(3,32)+R(3,2)*R(3,32)**2<R(21,50)
    assert R(1,2)-R(1,2)+R(3,2)*R(1,2)**2<R(21,50)
    assert 2*(R(4,15)-R(4,15)**2-R(3,100))/(1-R(4,15))**2<R(2,3)
    # The already proved quartic bound supplies the head deficit as well.
    assert R(2,3)<R(25,36)  # smaller binary probability exceeds 1/12
    assert s.expand(1-4*((1-z)/4)+R(3,8)*((1-z)/4)*z-(z+R(3,32)*z*(1-z)))==0
    assert 1+R(3,32)/3==R(33,32)
    # S weight: rho^(3/2) <= rho(rho+1/4) (AM-GM, sqrt(rho) <= rho+1/4), so
    # (1-u)^2(1+rho^q) <= (1-u)^2+u^2+u(1-u)/4 for rho=u/(1-u) <= 1.
    assert s.expand((t-R(1,2))**2-(t*t+R(1,4)-t))==0  # t=sqrt(rho): t <= t^2+1/4
    rho=u/(1-u)
    assert s.simplify((1-u)**2*(1+rho*(rho+R(1,4)))-((1-u)**2+u*u+u*(1-u)/4))==0
    assert s.expand(32*(1+2*u)*((1-u)**2+u*u+u*(1-u)/4)-(32+8*u*(1-7*u+14*u*u)))==0
    assert s.expand(1-7*u+14*u*u-(14*(u-R(1,4))**2+R(1,8)))==0
    assert 14*(R(1,8)-R(1,4))**2+R(1,8)==R(11,32) and abs(R(4,15)-R(1,4))<R(1,8)
    assert 8*R(1,8)*1<=1 and 8*R(4,15)*R(11,32)==R(11,15)<1
    # B weight: u^2((1-u)(3-2u))^5 <= 1 on [0,3/32]: increasing, endpoint below one.
    g=u**2*((1-u)*(3-2*u))**5
    assert s.cancel(s.diff(g,u)*u*(1-u)*(3-2*u)/g-(24*u*u-35*u+6))==0
    assert 24*R(3,32)**2-35*R(3,32)+6>0 and 35*R(3,32)<6
    assert (1-R(3,32))*(3-2*R(3,32))==R(1305,512)
    assert g.subs(u,R(3,32))<1
    assert -4+R(10,9)+R(5,7)<0
    eps,F=s.symbols('eps F')
    umax=R(47,100)**2+(R(7,25)+eps)**2+R(1,16)
    correction=R(11,10)*(R(3,4)*(1+eps)-R(3,8)-umax)
    assert s.expand(correction-R(11,10)*(R(33,2500)+R(19,100)*eps-eps*eps))==0
    excess=(1-20*eps-2*F)/9-correction
    assert excess.subs({eps:0,F:R(11,25)})==-R(89,75000)
    assert s.diff(excess,eps).subs(eps,R(1,100))<0
    print('CG-4 pair coordinates, binary curvature/weight identities and strict capped-cluster assembly PASS')


def negative_controls():
    spec=(F(121,250),)*2+(F(2,125),)*2
    p=sum(x*x for x in spec);third=sum(x**3 for x in spec);x=p-F(1,4)
    H=log_interval(sum((power(a,F(q)) for a in spec)))/((1-F(q))*LOG2)
    cb=F(10,3)-6*p+F(8,3)*third
    assert p<F(47,100)
    assert cb-F(H.lo,SCALE)-F(6,5)*x*(F(1,4)-x)<0
    # Removing the cap fails near flat-two, by the exact asymptotic slope.
    assert q/(r*LP)>R(149,40)
    print('CG-5 negative controls: coefficient 6/5 fails inside the cap; 11/10 fails without the cap PASS')


if __name__=='__main__':
    moments()
    binary_integral()
    binary_consequences()
    pairs_and_assembly()
    negative_controls()
