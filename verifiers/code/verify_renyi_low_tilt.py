#!/usr/bin/env python3
"""LT: algebra in the analytic repeated-largest chart reduction.

Proved in: renyi_low_tilt_reduction.tex. The divided-difference bound (DD),
feasibility of the path and its monotonicity are proved in the text. This
checks their generic identities, the elementary strict constant and two
negative controls.
"""
import sympy as s
from rational_intervals import LOG2, SCALE

R=s.Rational
u,d,x=s.symbols('u d x',positive=True)
a,b,c=s.symbols('a b c',positive=True)


def divided_difference_bound():
    # (DD): x^r - T is concave, T the cubic Taylor polynomial of x^r at the mean m.
    r,m,y=s.symbols('r m y',positive=True)
    dd=lambda g:g(a)/((a-b)*(a-c))+g(b)/((b-a)*(b-c))+g(c)/((c-a)*(c-b))
    mean=(a+b+c)/3
    assert s.simplify(dd(lambda z:(z-mean)**2)-1)==0
    assert s.simplify(dd(lambda z:(z-mean)**3))==0
    assert s.simplify(dd(lambda z:z)) == 0 and s.simplify(dd(lambda z:1+0*z)) == 0
    T=sum(s.diff(y**r,y,k).subs(y,m)*(y-m)**k/s.factorial(k) for k in range(4))
    second=s.diff(y**r-T,y,2)
    tangent=y**(r-2)-m**(r-2)-(r-2)*m**(r-3)*(y-m)
    assert s.simplify(second+r*(1-r)*tangent)==0
    # [a,b,c]T = (coefficient of (x-m)^2) = r(r-1)m^(r-2)/2.
    assert s.simplify(s.diff(T,y,2).subs(y,m)/2-r*(r-1)*m**(r-2)/2)==0
    print('LT-1 divided-difference bound (DD): [a,b,c](x-m)^2=1, [a,b,c](x-m)^3=0 and the concavity identity PASS')


def slope_bound():
    # r>=1/2, a>=1/4 and a(1-a)<=1/4 give the bound 3/8.
    assert R(9,16)*R(2,3)==R(3,8)
    assert 3**7>R(9,2)**5
    assert R(LOG2.hi,SCALE)<R(7,10)
    lower=R(16,25)/(6*R(7,10))*R(9,2)/R(3,8)
    assert lower==R(64,35)>R(9,5)
    assert 12*R(3,20)==R(9,5)
    print('LT-1 slope constant: eta > 64/35 > 9/5 on 3/2<=q<=8/5 PASS')


def path_identities():
    lam=(R(1,4)+u,)*2+(R(1,4)-u+d,R(1,4)-u-d)
    p=s.expand(sum(z*z for z in lam));t=s.expand(sum(z**3 for z in lam))
    assert p==R(1,4)+4*u*u+2*d*d
    path_t=R(1,16)+R(3,4)*x+12*u**3-3*x*u
    assert s.expand(t-path_t.subs(x,4*u*u+2*d*d))==0
    assert s.diff(path_t,u)==36*u*u-3*x
    assert s.expand((36*u*u-3*x).subs(x,4*u*u+2*d*d))==6*(4*u*u-d*d)
    dd=s.sqrt((x-4*u*u)/2)
    assert s.simplify(s.diff(dd,u)+2*u/dd)==0
    # Middle entry 1/4-u+d decreases; smallest entry 1/4-u-d has derivative 2u/d-1.
    assert s.simplify(s.diff(R(1,4)-u-dd,u)-(2*u/dd-1))==0
    assert s.simplify(s.diff(R(1,4)-u+dd,u)-(-1-2*u/dd))==0
    # The fixed-purity derivative ratio follows by quadratic interpolation.
    # These group velocities preserve trace and purity on (a,a,b,c).
    v=((b-c)/2,c-a,a-b)
    assert s.expand(2*v[0]+v[1]+v[2])==0
    assert s.expand(2*a*v[0]+b*v[1]+c*v[2])==0
    third=s.expand(2*a*a*v[0]+b*b*v[1]+c*c*v[2])
    assert s.factor(third)==(a-b)*(a-c)*(b-c)
    A,B,C=s.symbols('A B C')
    f=lambda z:A*z*z+B*z+C
    assert s.expand(2*f(a)*v[0]+f(b)*v[1]+f(c)*v[2]-A*third)==0
    print('LT-2 fixed-purity path, increasing third moment and divided-difference derivative identity PASS')


def negative_controls():
    # Flat-three and the equal-pair endpoint both have p=1/3.
    # At q=2 their entropies coincide, so a positive tilt decreases on the path.
    assert R(9,5)*(R(1,8)-R(1,9))==R(1,40)>0
    # A large tilt also defeats the reduction at orders in the stated q band.
    # Every A-pattern has nonnegative centred third moment, hence t>=1/8
    # at p=1/3. H<=2 then gives H-40t<=-3, but flat-three is greater.
    lamA=(R(1,4)+u+d,R(1,4)+u-d,R(1,4)-u,R(1,4)-u)
    assert s.expand(sum((z-R(1,4))**3 for z in lamA))==6*u*d*d
    assert 2-40*R(1,8)==-3
    assert 3**9>2**13  # log2(3)>13/9, so log2(3)-40/9>-3.
    print('LT-3 negative controls: extending to q=2 or removing the tilt bound fails PASS')


if __name__=='__main__':
    divided_difference_bound()
    slope_bound()
    path_identities()
    negative_controls()
