"""Three-copy spin-flip relaxation for any number of qubits n, reduced by Schur-Weyl duality for the site
symmetry (exact rational construction).

Statement: sec:threecopy-sw (in three_copy_schur_weyl.tex)  Imported by
certify_three_copy_schur_weyl_bounds.py, search_three_copy_sw_certificate.py, explore_three_copy_all_n.py.

Per site the U(2)^{(x)3} commutant on (C^2)^{(x)3} is End(S^[3]) + End(S^[21]), S^[3] = <|000>>,
S^[21] = <b1 = |100>-|010>, b2 = |010>-|001>> (spin-3/2 and spin-1/2 highest-weight vectors, Gram G0).
A test element g = (g0, g1), g1 a 2x2 matrix in the b-basis, is the commutant operator G with Fourier
blocks (g0/4, g1/2) (4, 2 = U(2) dimensions).  For the twirl Omega of (zz^*)^{(x)3}, symmetrised over
site permutations and z <-> conj(z), Schur-Weyl duality for GL(S^[21]) on the [21]-sites gives
  P(g) = tr[Omega G^{(x)n}] = sum_a C(n,a) g0^{n-a} sum_{nu=(a-q,q)} f_nu tr[M_{a,nu} W^T Gram_nu rho_nu(g1) W],
U_nu = Sym^{a-2q} (x) det^q in the monomial basis (Gram_nu its invariant Gram matrix), f_nu = dim of the
S_a irrep nu, W an integer basis of the S_3-invariant vectors (copy symmetry) and M_{a,nu} >= 0 real
symmetric: the omega-blocks, whose entries are the parameters.  The flipped moment (copy 3 replaced by
eps^{(x)n} conj z) has the same form with S_2-invariant bases and blocks N_{a,nu} >= 0 (gamma-blocks), and
generating polynomial P(T* g), T* the per-site flip on test elements.  The gamma-blocks are linear in
the parameters; exact_gamma_map finds the map by isolating each type a on rational points of the circle
(s, t) -> (s, t g1) and solving an exact square linear system per type.
"""
import itertools, math
from fractions import Fraction as Fr
import numpy as np
import flint

def Q(x, y=None):
    if y is not None: return flint.fmpq(x, y)
    if isinstance(x, Fr): return flint.fmpq(x.numerator, x.denominator)
    return flint.fmpq(x)
def ffloat(x): return float(Fr(int(x.p), int(x.q)))

# ---------------- per-site algebra (exact) ----------------
PERMS = list(itertools.permutations(range(3)))
def _op8(p):
    M = [[0]*8 for _ in range(8)]
    for idx in itertools.product(range(2), repeat=3):
        out = [0]*3
        for i in range(3): out[p[i]] = idx[i]
        M[out[0]*4+out[1]*2+out[2]][idx[0]*4+idx[1]*2+idx[2]] = 1
    return flint.fmpq_mat(M)
OP = [_op8(p) for p in PERMS]
B0 = flint.fmpq_mat([[1]]+[[0]]*7)
B21 = flint.fmpq_mat([[0, 0], [0, -1], [-1, 1], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0]])   # b1 = |100>-|010>, b2 = |010>-|001>
G0 = B21.transpose()*B21                                                                  # [[2,-1],[-1,2]]
_PINV = (B21.transpose()*B21).inv()*B21.transpose()
def fourier(G):
    """(f0, f1): G B0 = f0 B0, G B21 = B21 f1 for a commutant operator G."""
    f1 = _PINV*G*B21; assert G*B21 == B21*f1 and G*B0 == B0*(G*B0)[0, 0]
    return (G*B0)[0, 0], f1
RHO = [fourier(O)[1] for O in OP]
UD0, UD1 = 4, 2
def g_of(G):
    f0, f1 = fourier(G); return (UD0*f0, f1*UD1)
def _inv(k): p = PERMS[k]; q = [0]*3; [q.__setitem__(p[i], i) for i in range(3)]; return PERMS.index(tuple(q))
def G_of(g):
    """commutant operator with test-element coordinates g (Fourier inversion over S_3)."""
    g0, g1 = g; M = flint.fmpq_mat(8, 8)
    for k in range(6):
        Ri = RHO[_inv(k)]
        c = Q(g0)/UD0/6+sum((Q(2, 6)*Ri[j, i]*g1[i, j]/UD1 for i in range(2) for j in range(2)), Q(0))
        M += OP[k]*c
    return M
_E = [[0]*8 for _ in range(8)]
for _i in range(4): _E[2*_i][2*_i+1] = 1; _E[2*_i+1][2*_i] = -1
EPS3 = flint.fmpq_mat(_E)                     # I (x) I (x) eps
def flip(G):
    """(E^T G E) with the third tensor slot transposed."""
    M = EPS3.transpose()*G*EPS3; out = flint.fmpq_mat(8, 8)
    for a in range(8):
        for b in range(8): out[(a & 6) | (b & 1), (b & 6) | (a & 1)] = M[a, b]
    return out
def _tstar():
    cols = []
    for e in range(5):
        v = [0]*5; v[e] = 1
        h0, h1 = g_of(flip(G_of((Q(v[0]), flint.fmpq_mat([[v[1], v[2]], [v[3], v[4]]])))))
        cols.append([h0, h1[0, 0], h1[0, 1], h1[1, 0], h1[1, 1]])
    return flint.fmpq_mat(cols).transpose()
TSTAR = _tstar()
def tstar(g):
    v = flint.fmpq_mat([[g[0]], [g[1][0, 0]], [g[1][0, 1]], [g[1][1, 0]], [g[1][1, 1]]]); w = TSTAR*v
    return (w[0, 0], flint.fmpq_mat([[w[1, 0], w[2, 0]], [w[3, 0], w[4, 0]]]))
def check_site():
    """S_3 homomorphism, Fourier inversion, flip = involution up to sign, antisymmetriser = 0 on qubits."""
    for a in range(6):
        for b in range(6):
            c = PERMS.index(tuple(PERMS[a][PERMS[b][i]] for i in range(3)))
            assert OP[a]*OP[b] == OP[c] and RHO[a]*RHO[b] == RHO[c]
    for k in range(6): assert G_of(g_of(OP[k])) == OP[k]
    sgn = [1, -1, -1, 1, 1, -1]
    assert sum((OP[k]*sgn[k] for k in range(6)), flint.fmpq_mat(8, 8)) == flint.fmpq_mat(8, 8)
    for k in range(6): assert flip(flip(OP[k])) == OP[k]
    return True

# ---------------- GL(2) irreps in the monomial basis ----------------
def sym_rep(g1, p):
    """rho(g1) on Sym^p S, basis m_i = (orbit sum of) b1^{p-i} b2^i: entry (i,j) = [y^j] (g11+g12 y)^{p-i}(g21+g22 y)^i."""
    a = flint.fmpq_poly([g1[0, 0], g1[0, 1]]); b = flint.fmpq_poly([g1[1, 0], g1[1, 1]])
    rows = []
    for i in range(p+1):
        poly = a**(p-i)*b**i; rows.append([poly[j] for j in range(p+1)])
    return flint.fmpq_mat(rows)
def gram_sym(p):
    """Gram of the m_i under G0^{(x)p}: coefficient of x^{p-i}y^i x'^{p-j}y'^j in (2xx'-xy'-yx'+2yy')^p."""
    G = [[0]*(p+1) for _ in range(p+1)]
    for a, b, c in itertools.product(range(p+1), repeat=3):
        d = p-a-b-c
        if d < 0: continue
        G[c+d][b+d] += math.factorial(p)//(math.factorial(a)*math.factorial(b)*math.factorial(c)*math.factorial(d))*2**(a+d)*(-1)**(b+c)
    return flint.fmpq_mat(G)
def glrep(nu, g1):
    return sym_rep(g1, nu[0]-nu[1])*(g1.det()**nu[1])
def gram(nu): return gram_sym(nu[0]-nu[1])*(G0.det()**nu[1])
def fdim(a, q): return math.comb(a, q)-(math.comb(a, q-1) if q else 0)
def inv_basis(nu, group):
    """integer basis (rows) of the vectors of U_nu fixed by rho_nu(RHO[k]) for k in group."""
    import sympy as sp
    D = nu[0]-nu[1]+1; rows = []
    for k in group:
        R = glrep(nu, RHO[k])
        for i in range(D): rows.append([int(R[i, j].p)//int(R[i, j].q)-(1 if i == j else 0) for j in range(D)])
    out = []
    for v in sp.Matrix(rows).nullspace():
        v = v*sp.ilcm(1, 1, *[sp.fraction(x)[1] for x in v]); iv = [int(x) for x in v]
        g = math.gcd(*iv) or 1; out.append([x//g for x in iv])
    return out
S3 = list(range(6)); S2 = [k for k, p in enumerate(PERMS) if p[2] == 2]

class GammaMap(dict):
    """Dict bi -> list of parameter matrices, with packed block matrices S_blocks[bi] and S_nd[bi] for fast C contraction."""
    def __init__(self):
        super().__init__()
        self.S_blocks = {}; self.S_nd = {}

class Structure:
    """Blocks and parameters at n sites: blocks[name] = list of (a, nu, W rows, W^T fmpq, coef)."""
    def __init__(self, n):
        self.n = n; self.blocks = {}; self._gram = {}; self._wg = {}
        for name, grp in (('om', S3), ('ga', S2)):
            out = []; wcache = {}
            for a in range(n+1):
                for q in range(a//2+1):
                    nu = (a-q, q); key = (a-2*q, q % 2)
                    if key not in wcache:
                        W = inv_basis(nu, grp)
                        wcache[key] = (flint.fmpq_mat(W), flint.fmpq_mat(W).transpose()) if W else None
                    if wcache[key] is not None:
                        Wq, WTq = wcache[key]
                        out.append((a, nu, Wq, WTq, Q(math.comb(n, a)*fdim(a, q))))
            self.blocks[name] = out
            self._wg[name] = {key: ( val[0]*gram_sym(key[0]), val[1] ) for key, val in wcache.items() if val is not None}
        self.params = [(bi, k, l) for bi, (a, nu, W, WT, co) in enumerate(self.blocks['om']) for k in range(W.nrows()) for l in range(k, W.nrows())]
        self.NP = len(self.params)
    def size(self, name, bi): return self.blocks[name][bi][2].nrows()
    def gram(self, nu):
        if nu not in self._gram: self._gram[nu] = gram(nu)
        return self._gram[nu]
    def _kcache(self, name, g1):
        a = flint.fmpq_poly([g1[0, 0], g1[0, 1]]); b = flint.fmpq_poly([g1[1, 0], g1[1, 1]])
        ap = [flint.fmpq_poly([1])]; bp = [flint.fmpq_poly([1])]; z = Q(0)
        for _ in range(self.n): ap.append(ap[-1]*a); bp.append(bp[-1]*b)
        sym = {}; out = {}
        for (p, q2), (WG, WT) in self._wg[name].items():
            if p not in sym:
                rows = []
                for i in range(p+1):
                    c = (ap[p-i]*bp[i]).coeffs()
                    if len(c) <= p: c.extend([z]*(p+1-len(c)))
                    rows.append(c)
                sym[p] = flint.fmpq_mat(rows)
            out[p, q2] = WG*sym[p]*WT
        return out
    def Rlist(self, name, g):
        """per block: coef g0^(n-a) W Gram_nu rho_nu(g1) W^T, so that P(g) = sum_blocks tr[M R]."""
        g0, g1 = g; kc = self._kcache(name, g1); d = 3*g1.det()
        gp = [Q(1)]
        for _ in range(self.n): gp.append(gp[-1]*g0)
        dp = [Q(1)]
        for _ in range(self.n//2+1): dp.append(dp[-1]*d)
        return [kc[nu[0]-nu[1], nu[1] % 2]*(co*dp[nu[1]]*gp[self.n-a]) for (a, nu, W, WT, co) in self.blocks[name]]
    def P_params(self, g):
        g0, g1 = g; kc = self._kcache('om', g1); d = 3*g1.det()
        gp = [Q(1)]
        for _ in range(self.n): gp.append(gp[-1]*g0)
        dp = [Q(1)]
        for _ in range(self.n//2+1): dp.append(dp[-1]*d)
        kv = {key: [M[k, k] if k == l else M[k, l]+M[l, k] for k in range(M.nrows()) for l in range(k, M.nrows())] for key, M in kc.items()}
        out = []
        for (a, nu, W, WT, co) in self.blocks['om']:
            sc = co*dp[nu[1]]*gp[self.n-a]
            out.extend(v*sc for v in kv[nu[0]-nu[1], nu[1] % 2])
        return out
    def functionals(self, k):
        """exact normalisation tr Omega and mean k-body purity, as linear functionals on the parameters."""
        n = self.n; e = (Q(UD0), flint.fmpq_mat([[UD1, 0], [0, UD1]]))
        tau = g_of(OP[PERMS.index((1, 0, 2))])
        on = self.P_params(e)
        vals = [self.P_params((e[0]+z*tau[0], e[1]+tau[1]*z)) for z in range(n+1)]
        Vi = flint.fmpq_mat([[Q(z)**j for j in range(n+1)] for z in range(n+1)]).inv()
        oo = [sum((Vi[k, z]*vals[z][p] for z in range(n+1)), Q(0))/math.comb(n, k) for p in range(self.NP)]
        return on, oo

def circle_point(j, N):
    """rational point (s,t) on the unit circle near angle pi j/N."""
    m = Fr(math.tan(math.pi*j/N/2)).limit_denominator(1000)
    return Q((1-m*m)/(1+m*m)), Q(2*m/(1+m*m))

def exact_gamma_map(st, seed=5, log=None):
    """dict bi -> list over parameters p of the exact gamma block (fmpq_mat) for the unit parameter p."""
    n = st.n; bl = st.blocks['ga']; rng = np.random.default_rng(seed)
    bytype = {}
    for bi, blk in enumerate(bl): bytype.setdefault(blk[0], []).append(bi)
    E = {a: sum(st.size('ga', bi)*(st.size('ga', bi)+1)//2 for bi in bis) for a, bis in bytype.items()}
    circ = [(Q(3), Q(3*(j - n//2))) for j in range(n+1)]
    Vi_T = flint.fmpq_mat([[s**(n-a)*t**a for (s, t) in circ] for a in range(n+1)]).inv().transpose()
    g1s = []; seen = set()      # pairwise non-proportional integer matrices (rho_nu is homogeneous)
    while len(g1s) < max(E.values()):
        v = [int(x) for x in rng.integers(-40, 41, size=4)]; gg = math.gcd(*v)
        if gg == 0 or v[0]*v[3] == v[1]*v[2]: continue
        v = [x//gg for x in v]; s = next(x for x in v if x)
        if s < 0: v = [-x for x in v]
        if tuple(v) in seen: continue
        seen.add(tuple(v)); g1s.append(flint.fmpq_mat([[v[0], v[1]], [v[2], v[3]]]))
    # [s^{n-a} t^a] of P(T*(s, t g1_m)) via integer rays (3*T* is integral) and FLINT matrix multiplication
    B_by_m = [Vi_T*flint.fmpq_mat([st.P_params(tstar((s, g1*t))) for (s, t) in circ]) for g1 in g1s]
    Rv = [st.Rlist('ga', (Q(1), g1)) for g1 in g1s]
    if log: log('  gamma map: %d point evaluations done' % (len(g1s)*len(circ)))
    out = GammaMap()
    for a, bis in sorted(bytype.items()):
        cols = [(bi, i, k) for bi in bis for i in range(st.size('ga', bi)) for k in range(i, st.size('ga', bi))]
        A = flint.fmpq_mat([[Rv[m][bi][i, i] if i == k else Rv[m][bi][i, k]+Rv[m][bi][k, i] for (bi, i, k) in cols] for m in range(E[a])])
        B = flint.fmpq_mat([[B_by_m[m][a, p] for p in range(st.NP)] for m in range(E[a])])
        S = A.solve(B)
        offset = 0
        for bi in bis:
            sz = st.size('ga', bi); nc = sz*(sz+1)//2
            S_bi = flint.fmpq_mat([[S[offset+r, p] for p in range(st.NP)] for r in range(nc)])
            out.S_blocks[bi] = S_bi; out.S_nd[bi] = S_bi.numer_denom()
            offset += nc
            out[bi] = [flint.fmpq_mat(sz, sz) for _ in range(st.NP)]
        for c, (bi, i, k) in enumerate(cols):
            for p in range(st.NP): out[bi][p][i, k] = S[c, p]; out[bi][p][k, i] = S[c, p]
        if log: log('  gamma type %d: %d unknowns solved' % (a, E[a]))
    return out

def ldl(M):
    """exact M = L D L^T for a positive definite symmetric fmpq_mat: returns (L^{-1}, diag D)."""
    m = M.nrows(); A = [[M[i, j] for j in range(m)] for i in range(m)]; L = flint.fmpq_mat(m, m)
    for i in range(m):
        L[i, i] = 1
        for j in range(i+1, m):
            f = A[j][i]/A[i][i]; L[j, i] = f
            if f != 0:
                for k in range(i, m): A[j][k] -= f*A[i][k]
    return L.inv(), [A[i][i] for i in range(m)]

def gamma_multipliers(st, G, rec):
    """Exact gamma multipliers of a stored certificate: Z_b = L^{-T} Y'_b L^{-1}, where L D L^T is gamma block b
    at the rational centre rec['centre'] and Y'_b = rec['Y'][b] (float.hex entries, exact dyadic rationals)."""
    xc = [Q(Fr(v)) for v in rec['centre']]; xc_col = flint.fmpq_mat([[v] for v in xc]); Z = {}
    for b, rows in rec['Y'].items():
        bi = int(b); m = st.size('ga', bi)
        if hasattr(G, 'S_blocks') and bi in G.S_blocks:
            vec = G.S_blocks[bi]*xc_col; M0 = flint.fmpq_mat(m, m); idx = 0
            for i in range(m):
                for k in range(i, m): M0[i, k] = M0[k, i] = vec[idx, 0]; idx += 1
        else:
            arr = G[bi]
            M0 = sum((arr[p]*xc[p] for p in range(st.NP) if xc[p] != 0), flint.fmpq_mat(m, m))
        Li, D = ldl(M0); Y = flint.fmpq_mat([[Q(Fr(float.fromhex(v))) for v in r] for r in rows])
        assert Y == Y.transpose()
        Z[bi] = Li.transpose()*Y*Li
    return Z

def _tree_sum_nd(pairs):
    if len(pairs) == 1: return pairs[0]
    mid = len(pairs)//2
    N1, D1 = _tree_sum_nd(pairs[:mid]); N2, D2 = _tree_sum_nd(pairs[mid:])
    g = D1.gcd(D2); d1, d2 = D1//g, D2//g
    return N1*d2 + N2*d1, d1*D2

def gamma_adjoint(st, G, Z):
    """Exact dual contraction c_p = sum_b tr[Gamma_b(e_p) Z_b]."""
    if hasattr(G, 'S_nd') and Z:
        pairs = []
        for bi, M in Z.items():
            m = M.nrows()
            z_row = [M[i, i] if i == k else 2*M[i, k] for i in range(m) for k in range(i, m)]
            Nz, Dz = flint.fmpq_mat([z_row]).numer_denom()
            Ns, Ds = G.S_nd[bi]
            pairs.append((Nz*Ns, Dz*Ds))
        N_tot, D_tot = _tree_sum_nd(pairs)
        return [Q(N_tot[0, p], D_tot) for p in range(st.NP)]
    c = [Q(0)]*st.NP
    for b, M in Z.items():
        m = M.nrows()
        for p in range(st.NP):
            A = G[b][p]; c[p] += sum((A[i, j]*M[i, j] for i in range(m) for j in range(m) if A[i, j] != 0), Q(0))
    return c

def is_pd(M):
    """exact positive definiteness of a symmetric fmpq_mat (all LDL pivots positive)."""
    m = M.nrows(); A = [[M[i, j] for j in range(m)] for i in range(m)]
    for i in range(m):
        if A[i][i] <= 0: return False
        for j in range(i+1, m):
            f = A[j][i]/A[i][i]
            if f != 0:
                for k in range(i, m): A[j][k] -= f*A[i][k]
    return True

def omega_residual(st, on, oo, lam, c):
    """omega-block matrices of d = oo - lam*on - c (off-diagonal parameters split in half)."""
    d = [oo[p]-lam*on[p]-c[p] for p in range(st.NP)]; out = []
    for bi in range(len(st.blocks['om'])):
        m = st.size('om', bi); D = flint.fmpq_mat(m, m)
        for p, (b, k, l) in enumerate(st.params):
            if b == bi:
                if k == l: D[k, k] = d[p]
                else: D[k, l] = d[p]/2; D[l, k] = d[p]/2
        out.append(D)
    return out
