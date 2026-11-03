#!/usr/bin/env python3
"""Checks for the general envelope and for the direct cubic proof of (FGI).

Homes: FG-1--FG-5, FG-9: sup_flip_gradient.tex (supplement; proof body flip_gradient_proof.tex), the direct
proof of (FGI) of sup_general_envelope.tex#lem:fqe-flip-gradient (the point-evaluation proof is checked by
verify_flip_gradient_interpolation.py). FG-6--FG-8: the proof of thm:fqe-general-envelope (stated in
sextic_identity.tex, proved in sup_general_envelope.tex) and spin_flip_calibration.tex.
FG-1: Schouten substitutions make T=V-6U fully symmetric.
FG-2: copy-label Casimir selection rules, moment operators and the sector table on the full
      invariant multiplicities (dimension at most 16), no state evaluations.
FG-3: harmonic cubic scale 54 and the Klein-four quartic selection rule.
FG-4: existing (C), (U), (Pr) give (CY); polarization gives (FGI).
FG-5: exact nonzero coefficients for wrong-factor/omitted-correction controls.
FG-6: random full-space finite differences of (FG) at general z and the
      envelope (GE), numerical sanity only.
FG-7: twice the real part of (FGI) with the purity identity gives (FG); the
      pencil derivative, integrating factor, antiderivative and final
      substitution in the proof of thm:fqe-general-envelope.
FG-8: exact calibration derivative and closed site/pair norm integrals.
FG-9: exact spot checks, at Gaussian-integer states, of every intermediate
      statement of the text proof: (CP), (CY), the vanishing quartic
      covariant, the polarized u, q, d-y, and L = DH (regression only).
No normal form, generic block contraction table or determining points are used.
"""
# log: verify_envelope_flip_gradient.txt
from itertools import permutations
import numpy as np
import sympy as sp

res = []


def check(tag, ok, msg):
    res.append(bool(ok))
    print(('[PASS] ' if ok else '[FAIL] ') + tag + ' ' + msg)


EPS = np.array([[0, 1], [-1, 0]], dtype=object)
LOP = np.zeros((2, 2, 2, 2), dtype=object)   # L|xyt> = eps_yt |x>,   index [out, x, y, t]
MOP = np.zeros((2, 2, 2, 2), dtype=object)   # M|xyt> = eps_xt |y>
for a in range(2):
    for b in range(2):
        for c in range(2):
            LOP[a, a, b, c] += EPS[b, c]
            MOP[b, a, b, c] += EPS[a, c]
SYM6 = np.zeros((2,) * 6, dtype=object)       # 6 * symmetriser of three slots
for b in np.ndindex(2, 2, 2):
    for p in permutations(range(3)):
        SYM6[tuple(b[p[k]] for k in range(3)) + b] += 1
LET = 'abcdefghijkl'


def e4(t):
    return np.einsum('ae,bf,cg,dh,efgh->abcd', EPS, EPS, EPS, EPS, t)


def uv(x, y, t):
    """6U_i, 6V_i of w = x(x)y(x)t (not symmetrised); output legs in site order."""
    w = np.einsum('abcd,efgh,ijkl->aeibfjcgkdhl', x, y, t)   # site-major (x1,y1,t1,x2,...)
    out = []
    for i in range(4):
        pair = []
        for op in (LOP, LOP + MOP):
            spec, ops, outs = [], [], ''
            for s in range(4):
                a, b, c = LET[3 * s:3 * s + 3]
                if s == i:
                    o = chr(65 + 3 * s) + chr(66 + 3 * s) + chr(67 + 3 * s)
                    spec.append(o + a + b + c)
                    ops.append(SYM6)
                else:
                    o = chr(65 + 3 * s)
                    spec.append(o + a + b + c)
                    ops.append(op)
                outs += o
            pair.append(np.einsum(LET + ',' + ','.join(spec) + '->' + outs, w, *ops, optimize='greedy'))
        out.append(pair)
    return out



def eps_all(x):
    for leg in range(x.ndim):
        x = np.moveaxis(np.tensordot(EPS, x, axes=([1], [leg])), 0, leg)
    return x


def marg(z, w, keep):
    rest = [s for s in range(4) if s not in keep]
    zz = np.transpose(z, list(keep) + rest).reshape(2 ** len(keep), -1)
    ww = np.transpose(w, list(keep) + rest).reshape(2 ** len(keep), -1)
    return zz.dot(ww.T)


def tr(m):
    return sum((m[k, k] for k in range(m.shape[0])), 0)



def zero(expr):
    return sp.expand(expr) == 0


def bilinear(xs):
    return sp.expand(sum(np.sum(x * eps_all(x)) for x in xs))


def site_x(z, w):
    return [np.vectorize(lambda q: sp.expand(q / 6))(v - 8 * u)
            for u, v in uv(z, z, w)]


# Symmetrization in the first two inputs replaces U by (LLL+MMM)/2.
ll, mm = sp.symbols('l0:3'), sp.symbols('m0:3')
tensor = sp.prod(l + m for l, m in zip(ll, mm)) - 3 * (sp.prod(ll) + sp.prod(mm))
swap12 = dict(zip(ll + mm, mm + ll))
swap23 = dict(zip(ll + mm, tuple(-l for l in ll) + tuple(m-l for l, m in zip(ll, mm))))
check('FG-1', zero(tensor.xreplace(swap12) - tensor) and zero(tensor.xreplace(swap23) - tensor),
      'T=V-6U is fixed by both generating input transpositions (Schouten)')

from itertools import combinations
I=sp.I
II=sp.eye(2); sx=sp.Matrix([[0,1],[1,0]]);sy=sp.Matrix([[0,-I],[I,0]]);sz=sp.diag(1,-1)
cp=sp.Matrix([[-sp.Rational(1,2),-sp.sqrt(3)/2],[sp.sqrt(3)/2,-sp.Rational(1,2)]])
refs=[sz,cp*sz*cp.T,cp.T*sz*cp];group=[II,sz,cp,cp.T,sz*cp,sz*cp.T]
def simp(M):return M.applyfunc(sp.simplify)
def iszero(M):return all(sp.simplify(a)==0 for a in M)
def kron(ms):return sp.kronecker_product(*ms) if ms else sp.ones(1)
casok=momok=tabok=True
h=sp.Rational(1,2)
SECTORS={0:[(0,0,4,0,0,3,0)],2:[(1,0,0,0,2,5,0)],3:[(3*h,9,-2,0,-3*h,6,54)],
         4:[(2,0,-4,6,-2,4,0),(0,0,-4,0,-8,-47,0)]}
for k in range(5):
 eye=sp.eye(2**k); P=simp(sum((kron([g]*k) for g in group),sp.zeros(2**k))/6)
 def emb(mat,i):return kron([mat if j==i else II for j in range(k)])
 J2=simp(sum((sum((emb(q,j) for j in range(k)),sp.zeros(2**k))**2/4 for q in [sx,sy,sz]),sp.zeros(2**k)))
 Y2=simp(sum((emb(sy,j) for j in range(k)),sp.zeros(2**k))**2)
 plus=J2*P/6 if k==4 else sp.zeros(2**k)
 casok &= iszero(Y2*P-9*(P if k==3 else sp.zeros(2**k)))
 casok &= iszero((k*k-4*k)*P+2*J2*P-Y2*P/2-12*plus)
 def siteprod(mats):return kron(mats[:k])
 AA=sp.zeros(2**k);BB=AA.copy();Pi=AA.copy()
 for i,j in combinations(range(4),2):
  for r0 in refs:
   mats=[r0 if h in [i,j] else II for h in range(4)];AA+=siteprod(mats)
  for c0 in [cp,cp.T]:
   mats=[c0 if h in [i,j] else II for h in range(4)];BB+=siteprod(mats)
  for r0 in refs:
   for r1 in refs:
    if r0==r1:continue
    mats=[r0 if h==i else r1 if h==j else II for h in range(4)];Pi+=siteprod(mats)/6
 # Sites >= k carry trivial representations, so siteprod drops them.
 Hop=sp.Rational(17,6)*AA-3*BB-12*eye
 Hform=3*eye-sp.Rational(5,2)*k*(k+1)*eye+sp.Rational(17,2)*J2+Y2/8
 Cform=(2*k-k*k)*eye+J2-Y2/4;Cop=(12-sp.Rational(9,2)*k)*eye-2*Pi
 Wop=(4-2*k)*eye;tau=sp.zeros(2**k)
 if k==4:
  tau=sp.Rational(16,3)*sum((kron([(II-r0)/2]*4) for r0 in refs),sp.zeros(16))
 Aform=sp.Rational(3,2)*((4-k)*(3-k)-k)*eye+3*J2-sp.Rational(3,4)*Y2
 Bform=(12-sp.Rational(21,2)*k+sp.Rational(9,4)*k*k)*eye-sp.Rational(3,4)*Y2
 momok &= all(iszero(q) for q in [(AA-Aform)*P,(BB-Bform)*P,(Hop-Hform)*P,(Cop-Cform)*P,tau*P-6*plus])
 cnorm=54*(P if k==3 else sp.zeros(2**k))
 ynorm=(sp.Rational(8,3)*Hop-sp.Rational(2,3)*Wop-sp.Rational(28,3)*Cop)*P
 momok &= iszero(cnorm-ynorm-(sp.Rational(32,3)*(eye-tau)-4*Wop-sp.Rational(8,3)*Cop)*P)
 # The sector table of the supplement: rows (j, Y^2, NW1, N|s|^2, K_c, F, 54P_3) at this K.
 rows=SECTORS.get(k,[]);jv=[sp.Rational(a,2) for a in range(k%2,k+1,2)]
 tot=sp.zeros(2**k)
 for (j,y2,w,t,c,f,p3) in rows:
  Q=P
  for j2 in jv:
   if j2!=j:Q=Q*(J2-j2*(j2+1)*eye)/(j*(j+1)-j2*(j2+1))
  Q=simp(Q);tot+=Q
  tabok &= (not iszero(Q)) and all(iszero(op*Q-v*Q) for op,v in
   [(Y2,y2),(Wop,w),(tau,t),(Cform,c),(Hform,f),(cnorm,p3)])
 tabok &= iszero(tot-P)

check('FG-2', casok and momok and tabok,
      'copy-label selection rules, moment operators and cubic polarization defect on every invariant sector;'
      ' the sector table (K, j, Y^2) and its operator values are complete and exact')
# The harmonic cubic intertwiner, including its normalization.
lrow=sp.Matrix([[sp.sqrt(sp.Rational(3,2)),-1/sp.sqrt(2)]])
mrow=sp.Matrix([[sp.sqrt(sp.Rational(3,2)),1/sp.sqrt(2)]])
trow=kron([lrow+mrow]*3)-3*(kron([lrow]*3)+kron([mrow]*3))
pthree=sum((kron([g]*3) for g in group),sp.zeros(8))/6
highest=sp.zeros(8,2)
highest[1,0]=2/sp.sqrt(6);highest[2,0]=highest[4,0]=-1/sp.sqrt(6)
highest[2,1]=1/sp.sqrt(2);highest[4,1]=-1/sp.sqrt(2)
ok3=iszero(sp.Matrix(LOP.reshape(2,8))*highest-lrow.col_join(sp.zeros(1,2)))
ok3 &= iszero(sp.Matrix(MOP.reshape(2,8))*highest-mrow.col_join(sp.zeros(1,2)))
ok3 &= iszero(trow.T*trow-54*pthree)
# Klein four acts trivially on the four-spin singlets, and has no fixed
# vector on the standard traceless representation [31] of S4.
klein=[(0,1,2,3),(1,0,3,2),(2,3,0,1),(3,2,1,0)]
E=np.array([[0,1],[-1,0]],dtype=object)
singlets=[np.einsum('ab,cd->abcd',E,E),np.einsum('ac,bd->abcd',E,E)]
projstd=sp.eye(4)-sp.ones(4)/4
avg=sp.zeros(4)
for perm in klein:
 mat=sp.zeros(4)
 for j in range(4):mat[perm[j],j]=1
 avg+=mat/4
 ok3 &= all(np.array_equal(q.transpose(perm),q) for q in singlets)
ok3 &= iszero(avg*projstd)
check('FG-3', ok3, 'harmonic cubic has squared scale 54; Klein-four selection forbids the quartic spin-one covariant')

# The existing contraction identities (C), (U), (Pr) give (CY).
N0,S10,P0,T0,Pi0=sp.symbols('N S1 P T Pi')
W0x=2*S10-4*N0**2;C0x=sp.Rational(9,2)*N0*S10-6*N0**3-2*Pi0
F0=17*N0*P0-12*T0-12*N0**3
uu0=(2*N0*S10-4*N0**3)/3
uv0=(-8*N0**3-2*N0*S10+8*Pi0)/3
vv0=(128*N0**3-226*N0*S10+136*N0*P0+152*Pi0)/3-32*T0
ok4=zero(vv0-12*uv0+36*uu0-(sp.Rational(8,3)*F0-sp.Rational(2,3)*N0*W0x-sp.Rational(28,3)*C0x))
ok4 &= zero(uv0-6*uu0-(sp.Rational(2,3)*N0*W0x-sp.Rational(4,3)*C0x))
s0,n0,sb0,W10,p0,y0=sp.symbols('s n sb W1 p y')
u0=-s0*W10/3;q0=s0*W10/3+2*p0/3
d0=y0+sp.Rational(16,3)*s0*(n0*n0-s0*sb0)-2*s0*W10+4*p0/3
L0=sp.Rational(3,8)*(d0+2*y0-8*q0-4*u0)-sp.Rational(3,2)*p0
B0=y0-4*q0+4*u0
correction=s0*(2*n0*n0-2*s0*sb0+sp.Rational(7,4)*W10)
ok4 &= zero(L0-sp.Rational(9,8)*B0-correction)
check('FG-4',ok4,'mixed cubic norm identities and one polarization give (FGI), with all mismatch terms cancelling')
check('FG-5',sp.expand(L0-B0-correction).coeff(y0)==sp.Rational(1,8)
      and sp.expand(L0-sp.Rational(9,8)*B0).coeff(n0,2)==2*s0,
      'negative controls: wrong factor leaves y/8; omitted correction leaves 2s n^2')

# Full-space sanity only; proof checks above are exact representation and polynomial identities.
def h_numeric(z):
    zz = z.conj()
    nn = float(np.vdot(z.ravel(), z.ravel()).real)
    pp = tt = ww = 0.
    for keep in [(0, 1), (0, 2), (0, 3)]:
        R = np.asarray(marg(z, zz, keep), dtype=complex)
        pp += np.trace(R@R).real
        tt += np.trace(R@R@R).real
    for j in range(4):
        R = np.asarray(marg(z, zz, [j]), dtype=complex)
        ww += 2*np.trace(R@R).real-nn*nn
    return 17*nn*pp-12*tt-12*nn**3-.75*nn*ww


SY4 = np.kron(np.kron([[0, -1j], [1j, 0]], [[0, -1j], [1j, 0]]),
              np.kron([[0, -1j], [1j, 0]], [[0, -1j], [1j, 0]]))
PAULI = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]


def w1_numeric(z):
    out = 0.
    for i in range(4):
        m = np.moveaxis(z, i, 0).reshape(2, 8)
        r = m @ m.conj().T
        out += sum(abs(np.trace(r @ q))**2 for q in PAULI)
    return out


rng = np.random.default_rng(20261029)
err = gap = 0.
for _ in range(12):
    znum = (rng.normal(size=16)+1j*rng.normal(size=16)).reshape([2]*4)
    flip = np.asarray(e4(znum.conj()), dtype=complex)
    xs = [np.asarray((vv-8*uu)/6, dtype=complex) for uu, vv in uv(znum, znum, flip)]
    overlap = sum(np.sum(x*np.asarray(eps_all(x), dtype=complex)) for x in xs)
    nn = float(np.vdot(znum.ravel(), znum.ravel()).real)
    sval = znum.ravel() @ SY4 @ znum.ravel()
    w1 = w1_numeric(znum)
    step = 1e-5
    deriv = (h_numeric(znum+step*flip)-h_numeric(znum-step*flip))/(2*step)
    pred = 2.25*overlap.real+sval.real*(4*nn*nn-4*abs(sval)**2+3.5*w1)
    err = max(err, abs(deriv-pred)/nn**3)
    gap = min(gap, (h_numeric(znum)-nn*abs(sval)**2+abs(sval)**3
                    - 7*abs(sval)**2*w1/(8*(nn+abs(sval))))/nn**3)
check('FG-6', err < 1e-6 and gap > -1e-9,
      'numerical sanity only: (FG) at general z by finite differences, and (GE) at random states')

# FG-7: from (FGI) to (FG), and the integration in the proof of the envelope.
Nr, Wr, x, y = sp.symbols('N W1 x y', real=True)
sc = x+I*y
Pp = Nr**2+(Wr+x*x+y*y)/2
fgi = sc*(sp.Rational(5, 4)*Nr**2+sp.Rational(11, 8)*Wr+sp.Rational(3, 4)*Pp-sp.Rational(19, 8)*(x*x+y*y))
ok7 = zero(2*sp.re(sp.expand(fgi))-x*(4*Nr**2-4*(x*x+y*y)+sp.Rational(7, 2)*Wr))
tt, kk, W0, hh, hp = sp.symbols('t kappa W0 h hp', real=True)
al, be = sp.sqrt((1+tt)/2), sp.sqrt((1-tt)/2)        # z_t = al a + i be b, flip: al a - i be b
ok7 &= sp.simplify(2*(1-tt**2)*sp.diff(al, tt)-(1-tt)*al) == 0       # a-coefficient of flip - t z
ok7 &= sp.simplify(2*(1-tt**2)*sp.diff(be, tt)+(1+tt)*be) == 0       # b-coefficient: -be - t be
ok7 &= sp.simplify(4*al**2*be**2-(1-tt**2)) == 0                      # W1(z_t) = (1-t^2) W0
rhs = -6*(1+tt)*hh+4*tt*(1-tt**2)+sp.Rational(7, 2)*tt*(1-tt**2)*W0     # lower bound for 2(1-t^2)h'
ok7 &= sp.simplify((rhs/(2*(1-tt**2))+3*hh/(1-tt))/(1-tt)**3-(2+sp.Rational(7, 4)*W0)*tt/(1-tt)**3) == 0
ok7 &= sp.simplify(sp.integrate(tt/(1-tt)**3, (tt, 0, kk))-kk**2/(2*(1-kk)**2)) == 0
lower = (1-kk)**3*(kk**2/(1-kk)**2)*(1+sp.Rational(7, 8)*W0)
ok7 &= sp.simplify(lower.subs(W0, Wr/(1-kk**2))-(kk**2-kk**3+7*kk**2*Wr/(8*(1+kk)))) == 0
check('FG-7', ok7, '(FGI) + purity gives (FG); pencil derivative, W1 scaling, integrating factor and envelope algebra')
# FG-8: calibration transport and its closed sum of squared norms.
xx, br, cc, aa8, ae8, ee8, dd8, rr8 = sp.symbols('X2 ReB C2 A2 AE E2 D r', real=True)
hh8 = sp.Rational(3, 8)*xx+sp.Rational(3, 2)*cc
flipder = sp.Rational(9, 4)*br+4*tt*(1-tt**2)+sp.Rational(7, 2)*tt*(1-tt**2)*W0
mp = (flipder-6*tt*hh8)/(2*(1-tt**2)*(1-tt)**3)+3*hh8/(1-tt)**4 \
    -(2+sp.Rational(7, 4)*W0)*tt/(1-tt)**3
mpform = sp.Rational(9, 16)*(2*(xx+br)+8*cc)/((1-tt)**4*(1+tt))
ok8 = sp.simplify(mp-mpform) == 0
R8 = (1+kk)/(1-kk)
AB8, BB8 = sp.sqrt(2)*ae8-aa8, 2*ee8-2*sp.sqrt(2)*ae8+aa8
siteint = sp.Rational(9, 64)*(1-kk)**3*sp.integrate(rr8**2*aa8+2*rr8*AB8+BB8, (rr8, 1, R8))
siteform = sp.Rational(9, 32)*kk*(kk**2*aa8+2*sp.sqrt(2)*kk*(1-kk)*ae8+2*(1-kk)**2*ee8)+sp.Rational(3, 32)*kk**3*aa8
ok8 &= sp.simplify(siteint-siteform) == 0
pairint = sp.Rational(9, 2)*(1-kk)**3*sp.integrate((cc-tt*dd8)/(1-tt)**3, (tt, 0, kk))
pairform = sp.Rational(9, 2)*kk*(1-kk)**2*cc+sp.Rational(9, 8)*kk**2*(1-kk)*2*(cc-dd8)
ok8 &= sp.simplify(pairint-pairform) == 0
check('FG-8', ok8, 'exact calibration derivative and both closed positive-norm integrals (SC-SOS)')
# FG-9: exact Gaussian-integer spot checks of the text's intermediate identities.
# All tensors are integer-scaled (6U, 6V from uv), so float64 sums are exact below 2^52.
def gi(q):
    re_, im_ = float(np.real(q)), float(np.imag(q))
    assert abs(re_) < 2**52 and abs(im_) < 2**52 and re_ == round(re_) and im_ == round(im_)
    return sp.Integer(int(round(re_)))+I*sp.Integer(int(round(im_)))


def cx(a):
    return np.asarray(a, dtype=complex)


LOPc, MOPc, SYM6c, EPSc = cx(LOP), cx(MOP), cx(SYM6), cx(EPS)


def uvc(x, y, t):
    w = np.einsum('abcd,efgh,ijkl->aeibfjcgkdhl', x, y, t)
    out = []
    for i in range(4):
        pair = []
        for op in (LOPc, LOPc+MOPc):
            spec, ops, outs = [], [], ''
            for s_ in range(4):
                a_, b_, c_ = LET[3*s_:3*s_+3]
                o = chr(65+3*s_)+chr(66+3*s_)+chr(67+3*s_) if s_ == i else chr(65+3*s_)
                spec.append(o+a_+b_+c_)
                ops.append(SYM6c if s_ == i else op)
                outs += o
            pair.append(np.einsum(LET+','+','.join(spec)+'->'+outs, w, *ops, optimize='greedy'))
        out.append(pair)
    return out


def u12(x, y, t):   # 12 U_i symmetrized in the first two inputs
    return [a[0]+b[0] for a, b in zip(uvc(x, y, t), uvc(y, x, t))]


def t6(x, y, t):    # 6 T_i = 6V_i - 36 U_i(sym)
    return [a[1]-3*b for a, b in zip(uvc(x, y, t), u12(x, y, t))]


def e4c(t):
    return np.einsum('ae,bf,cg,dh,efgh->abcd', EPSc, EPSc, EPSc, EPSc, t)


def bil(v, w):
    for leg in range(w.ndim):
        w = np.moveaxis(np.tensordot(EPSc, w, axes=([1], [leg])), 0, leg)
    return np.sum(v*w)


def moms(z, zb):
    nn = np.sum(z*zb)
    pp = tt3 = 0
    for keep in [(0, 1), (0, 2), (0, 3)]:
        r = cx(marg(z, zb, keep))
        pp += np.trace(r@r)
        tt3 += np.trace(r@r@r)
    rs = [cx(marg(z, zb, [i])) for i in range(4)]
    ww = sum(2*np.trace(r@r)-nn*nn for r in rs)
    return nn, pp, tt3, ww, rs


def site_apply(r, x, i):
    return np.moveaxis(np.tensordot(r, x, axes=([1], [i])), 0, i)


ok9 = True
rng9 = np.random.default_rng(9)
for _ in range(2):
    z9 = cx(rng9.integers(-1, 2, 16)+1j*rng9.integers(-1, 2, 16)).reshape([2]*4)
    w9 = cx(rng9.integers(-1, 2, 16)+1j*rng9.integers(-1, 2, 16)).reshape([2]*4)
    zb, zt = z9.conj(), e4c(z9.conj())
    nn, pp, tt3, ww, rs = moms(z9, zb)
    cs = [site_apply(rs[i], z9, i)-site_apply(rs[j], z9, j) for i, j in combinations(range(4), 2)]
    N9, W9, F9 = gi(nn), gi(ww), gi(17*nn*pp-12*tt3-12*nn**3)
    s9, sb9, n9 = gi(np.sum(z9*e4c(z9))), gi(np.sum(zt*e4c(zt))), gi(np.sum(z9*e4c(zt)))
    Kc9, p9 = gi(sum(np.sum(abs(c)**2) for c in cs)), gi(sum(np.sum(c*e4c(c)) for c in cs))
    C6, Y6, U12 = t6(z9, z9, z9), t6(z9, z9, zt), u12(z9, z9, zt)
    nC = gi(sum(np.sum(abs(c)**2) for c in C6))
    nY = gi(sum(np.sum(abs(c)**2) for c in Y6))
    nU = gi(sum(np.sum(abs(c)**2) for c in U12))
    UY = gi(sum(np.sum(np.conj(a)*b) for a, b in zip(U12, Y6)).real)
    ok9 &= zero(nC-nY-(384*N9*(N9**2-s9*sb9)-144*N9*W9-96*Kc9))                       # 36 (CP)
    ok9 &= zero(nY-(96*F9-24*N9*W9-336*Kc9)) and zero(nU-48*N9*W9) and zero(UY-(48*N9*W9-96*Kc9))  # (CY)
    ok9 &= gi(sum(bil(c, u) for c, u in zip(C6, u12(z9, z9, w9)))) == 0                 # quartic covariant
    u9 = gi(sum(bil(a, a) for a in U12))
    q9 = gi(sum(bil(a, b) for a, b in zip(Y6, U12)))
    y9 = gi(sum(bil(a, a) for a in Y6))
    d9 = gi(sum(bil(a, b) for a, b in zip(C6, t6(z9, zt, zt))))
    ok9 &= zero(u9+48*s9*W9) and zero(q9-(24*s9*W9+48*p9))                             # 144u, 72q
    ok9 &= zero(d9-y9-(192*s9*(n9**2-s9*sb9)-72*s9*W9+48*p9))                           # 36(d-y)
    h4 = []
    for k in (-2, -1, 1, 2):                    # 4H is cubic in the conjugate slot: exact stencil
        nk, pk, tk, wk, _ = moms(z9, zb+k*e4c(z9))
        h4.append(gi(4*(17*nk*pk-12*tk-12*nk**3)-3*nk*wk))
    L9 = (h4[0]-8*h4[1]+8*h4[2]-h4[3])/48
    ok9 &= zero(L9-(sp.Rational(3, 8)*(d9/36+2*y9/36-8*q9/72-4*u9/144)-sp.Rational(3, 2)*p9))
check('FG-9', ok9, 'exact Gaussian-integer checks of (CP), (CY), the quartic covariant, u, q, d-y and L = DH')
print('ALL PASS' if all(res) else 'FAILURES')
raise SystemExit(0 if all(res) else 1)
