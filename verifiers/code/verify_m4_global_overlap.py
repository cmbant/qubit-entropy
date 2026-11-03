#!/usr/bin/env python3
"""Exact rational normal-slice Positivstellensatz certificate for Q >= 1 - Phi.

Reconstructs every HS ray, matrix constraint, signed Reynolds coefficient and
positive rational Gram matrix on the 19-variable closest-point normal slice,
verifying the polynomial identity for (Q - n^2 e) / 36 over Q.
"""
import json
from fractions import Fraction as F
from itertools import combinations, permutations
from pathlib import Path
from sympy import symbols, expand
from orbit_certificate_polynomials import (
    PR, XS, ZERO, X, Y, n, Q, h0, e,
    defects, height_hessian, lower_frame, CHARS,
)

H = height_hessian()
root = Path(__file__).resolve().parent / 'data'
data = json.loads((root / 'm4_global_overlap.json').read_text())

N = data['variables']
assert N == 19 and data['which'] == 'Q_overlap'
assert data['relation'] == 'e=n-h0^2/6'
assert F(*data['coefficient']) == F(1, 1)

md, grid, cd, bits = (data[key] for key in ('map_den', 'grid', 'correction_den', 'factor_bits'))
assert all(isinstance(q, int) and q > 0 for q in (md, grid, cd, bits))
md *= 36  # clear the common basis denominator once


def pack(m):
    """Base-eight exponent digits; total degree <=6 makes products carry-free."""
    assert len(m) == N and all(isinstance(a, int) and a >= 0 for a in m)
    assert sum(m) <= 6
    return sum(a << (3*i) for i, a in enumerate(m))


def terms(poly):
    return {pack(tuple(m[:N])): F(int(c.numerator), int(c.denominator))
            for m, c in poly.items()}


rays = []
for real, imag in data['rays']:
    a, b = [[[PR(F(*v)) for v in row] for row in M] for M in (real, imag)]
    nn, qq, rr, ww, tt = defects(a, b)
    assert nn == 6 and qq == 0 and ww == 0 and tt == 0
    re = (3 * sum((a[i][j] * X[i][j] for i in range(4) for j in range(4)), ZERO)
          + sum((b[i][j] * Y[i][j] for i in range(4) for j in range(4)), ZERO))
    im = sum((a[i][j] * Y[i][j] - b[i][j] * X[i][j] for i in range(4) for j in range(4)), ZERO)
    rays.append((h0 * h0 - re * re - 3 * im * im) / 36)
print('[PASS] GO-1', N, 'variables:', len(rays), 'independently checked HS rays', flush=True)

acts, pairs = [], list(combinations(range(3), 2))
for pp in permutations(range(3)):
    mi, sg = list(range(N)), [1] * N
    for j in range(3):
        for offset in (1, 4, 13, 16):
            mi[offset + j] = offset + pp[j]
    for k, (i, j) in enumerate(pairs):
        ii, jj = pp[i], pp[j]
        kk = pairs.index(tuple(sorted((ii, jj))))
        mi[7 + k], mi[10 + k], sg[10 + k] = 7 + kk, 10 + kk, 1 if ii < jj else -1
    acts.append((mi, sg))

orbit_cache = {}


def transform(m, act):
    mi, sg = act
    out, sign = 0, 1
    for i in range(N):
        a = (m >> (3*i)) & 7
        out += a << (3*mi[i])
        if a & 1:
            sign *= sg[i]
    return out, sign


def orbit(m):
    if m in orbit_cache:
        return orbit_cache[m]
    ch = 0
    for i, c in enumerate(CHARS):
        if (m >> (3*i)) & 1:
            ch ^= c
    if ch:
        orbit_cache[m] = None
        return None
    values = {}
    for act in acts:
        mm, sign = transform(m, act)
        if mm in values and values[mm] != sign:
            orbit_cache[m] = None
            return None
        values[mm] = sign
    rep = min(values)
    for mm, sign in values.items():
        orbit_cache[mm] = (rep, sign * values[rep])
    return orbit_cache[m]



def unpack_gram(block):
    """Integer reconstruction and PSD proof; no eigensolver or floating arithmetic.

    G=(round(grid*A*A^T/scale^2)+Delta+C/cd)/grid.
    Each rounding error is at most 1/2 in grid units. Thus the row test
    below proves G-A*A^T/scale^2 strictly diagonally dominant and positive.
    """
    size = len(block['basis'])
    factor = block['factor']
    assert len(factor) == size
    rank = len(factor[0])
    assert all(len(row) == rank for row in factor)
    assert all(isinstance(v, int) for row in factor for v in row)
    scale_squared = 1 << (2 * bits)
    residual = block['residual_upper']
    assert len(residual) == size * (size + 1) // 2
    cor = dict(block['corrections'])
    assert len(cor) == len(block['corrections'])
    assert all(0 <= k < len(residual) and isinstance(v, int) for k, v in cor.items())
    upper, diagonal, offsum, offset = [], [0] * size, [0] * size, 0
    for j in range(size):
        for i in range(j + 1):
            product = sum(a * b for a, b in zip(factor[i], factor[j]))
            rounded = (grid * product + scale_squared // 2) // scale_squared
            delta = residual[offset]
            assert isinstance(delta, int)
            upper.append(rounded + delta)
            value = delta * cd + cor.get(offset, 0)
            if i == j:
                diagonal[i] = value
            else:
                offsum[i] += abs(value)
                offsum[j] += abs(value)
            offset += 1
    # One diagonal error and size-1 off-diagonal errors, each <= cd/2.
    assert all(2 * (a - b) > size * cd for a, b in zip(diagonal, offsum))
    return upper


constraints, constraint_degrees, accum, correction_accum = {}, {}, {}, {}
for block in data['blocks']:
    base_upper = unpack_gram(block)
    key = json.dumps(block['constraint'])
    if key not in constraints:
        tag, *args = block['constraint']
        if tag == 'sos':
            M = [[PR.one]]
        elif tag == 'fidelity':
            M = [[rays[args[0]]]]
        elif tag == 'lead':
            M = [[h0 * XS[0] / 6]]
        elif tag == 'excess':
            M = [[e / 6]]
        elif tag == 'height':
            M = [[n * h0 * z / 216 for z in row] for row in H]
        elif tag == 'imaginary':
            M = [[n * h0 * Y[i][j] / 36 for j in range(1, 4)] for i in range(1, 4)]
        elif tag == 'height_quadratic':
            M = [[h0 * z / 216 for z in row] for row in H]
        elif tag == 'imaginary_quadratic':
            M = [[h0 * Y[i][j] / 36 for j in range(1, 4)] for i in range(1, 4)]
        elif tag == 'procrustes':
            assert args[0] in (0, 1)
            K = [[3 * X[i][j] if (j == 0 if args[0] == 0 else i == 0) else Y[i][j]
                  for j in range(4)] for i in range(4)]
            assert K == [list(row) for row in zip(*K)]
            M = [[h0 * z / 216 for z in row] for row in K]
        elif tag == 'fidelity_floor':
            sr = (3 * sum((z * z for row in X for z in row), ZERO)
                  - sum((z * z for row in Y for z in row), ZERO))
            si = 2 * sum((a * b for row, rr in zip(X, Y) for a, b in zip(row, rr)), ZERO)
            ell = h0 * h0 - n
            M = [[(ell + sr) / 36, si / 36], [si / 36, (ell - sr) / 108]]
        elif tag == 'lower_quadratic':
            assert args in [[-1,-1,[-1,2],[1,2]], [-1,1,[1,2],[1,2]],
                            [1,-1,[-1,1],[0,1]], [1,1,[1,1],[0,1]]]
            a, b, c, s = args
            assert a in (-1, 1) and b in (-1, 1)
            M = [[h0 * z / 216 for z in row]
                 for row in lower_frame(a, b, PR(F(*c)), PR(F(*s)))]
        else:
            raise ValueError(f'Unknown constraint tag: {tag}')
        constraint_degrees[key] = max((sum(m) for row in M for z in row for m in z), default=0)
        scaled = []
        for row in M:
            parsed_row = []
            for z in row:
                entries = []
                for monomial, coefficient in terms(z).items():
                    coefficient *= F(md, 36)
                    assert coefficient.denominator == 1
                    entries.append((monomial, coefficient.numerator))
                parsed_row.append(entries)
            scaled.append(parsed_row)
        constraints[key] = scaled
    M = constraints[key]
    basis = []
    for row in block['basis']:
        parsed = []
        for component, num, den, inds in row:
            assert 0 <= component < len(M) and all(0 <= i < N for i in inds)
            mm = [0] * N
            for i in inds:
                mm[i] += 1
            assert 6 % den == 0
            parsed.append((component, pack(tuple(mm)), num * (6 // den)))
        basis.append(parsed)
    basis_degree = max((len(inds) for row in block['basis'] for _, _, _, inds in row), default=0)
    assert 2 * basis_degree + constraint_degrees[key] <= 6
    size, offset = len(basis), 0
    cor = {i: int(c) for i, c in block['corrections']}
    for j in range(size):
        for i in range(j + 1):
            value = base_upper[offset]
            correction = cor.get(offset, 0)
            offset += 1
            value *= 1 if i == j else 2
            correction *= 1 if i == j else 2
            for ai, a, ca in basis[i]:
                for bi, b, cb in basis[j]:
                    ab = a + b
                    for m, c in M[ai][bi]:
                        mm = ab + m
                        oo = orbit(mm)
                        if oo is None:
                            continue
                        rep, sign = oo
                        coeff = sign * c * ca * cb
                        accum[rep] = accum.get(rep, 0) + value * coeff
                        if correction:
                            correction_accum[rep] = correction_accum.get(rep, 0) + correction * coeff
    assert offset == len(base_upper)

# All 96 linear lower-frame blocks have an explicit analytic replacement.
sqsum = sum((z*z for z in XS[7:19]), ZERO)
differences = sum(((XS[i]-XS[j])**2 + (XS[i+3]-XS[j+3])**2
                   for i, j in combinations(range(1, 4), 2)), ZERO)
floor_term = F(1, 10**7)*n*h0*h0*(5*sqsum+2*differences)/27
axial = [XS[12], -XS[11], XS[10]]
Lstar = lower_frame(-1, -1, PR(F(-1, 2)), PR(F(1, 2)))
axial_term = F(794424, 10**12)*n*h0*sum(
    (axial[i]*Lstar[i][j]*axial[j] for i in range(3) for j in range(3)), ZERO)/216
for remainder in (floor_term, axial_term):
    rr = terms(remainder)
    assert all({transform(m, act)[0]: c*transform(m, act)[1] for m, c in rr.items()} == rr
               for act in acts)
target = terms((Q - n * n * e) / 36 - floor_term - axial_term)
assert all(orbit(m) is not None or not c for m, c in target.items())
for act in acts:
    transformed = {transform(m, act)[0]: c * transform(m, act)[1] for m, c in target.items()}
    assert transformed == target
expected = {}
for m, c in target.items():
    if not c:
        continue
    rep, sign = orbit(m)
    coeff = c * md * grid
    assert coeff.denominator == 1
    expected[rep] = expected.get(rep, 0) + sign * coeff.numerator
keys = expected.keys() | accum.keys() | correction_accum.keys()
assert all(correction_accum.get(m, 0) == cd * (expected.get(m, 0) - accum.get(m, 0)) for m in keys)
print('[PASS] GO-2 packed monomials and separate integer base/correction identities;',
      len(keys), 'coefficient keys;', max(abs(v).bit_length() for v in accum.values()),
      'maximum main-sum bits', flush=True)
print('[PASS] GO-2 compact factors:', sum(len(b['factor'][0]) for b in data['blocks']),
      'columns;', sum(len(row) for b in data['blocks'] for row in b['factor']),
      'factor entries;', sum(not len(b['factor'][0]) for b in data['blocks']), 'blocks without factors', flush=True)
print('[PASS] GO-2', N, 'variables:', len(data['blocks']),
      'positive rational Gram blocks plus two analytic remainders; exact identity for (Q - n^2 e)/36', flush=True)

# Fidelity floor (ordered real singular values m >= x >= y >= z): six times the second gap is a sum of
# nonnegative terms; then the exact transfer from Q >= 1 - Phi and Phi >= 1/6 to Q >= ((6+sqrt(6))/12) d^2.
sm, sx, sy, sz, u = symbols('m x y z u', real=True)
assert expand(sm**2 + (sx + sy + sz)**2 - 2 * (sx**2 + sy**2 + sz**2)
              - ((sm**2 - sx**2) + sy * (2 * sx - sy) + sz * (2 * sx - sz) + 2 * sy * sz)) == 0
assert (F(1, 2) + F(1, 6)) / 2 == F(1, 3) == 2 * F(1, 6)
assert expand((1 - u**2) - ((1 + u) / 2) * 2 * (1 - u)) == 0
assert F(2, 5)**2 < F(1, 6) and (1 + F(2, 5)) / 2 == F(7, 10)
print('[PASS] GO-3 fidelity-floor remainder; Q >= 1 - Phi >= ((6+sqrt(6))/12) * distance^2 > (7/10) * distance^2',
      flush=True)


# Theorem fqe-orbit-stability: delta >= k Q >= k (6+sqrt(6))/12 d^2 > d^2/20, k = (ln 3 - 1)/(2 ln 2).
def log_bounds(w):
    """Rational bounds lo < ln((1+w)/(1-w)) < hi from the atanh series, 0 < w < 1."""
    lo = 2 * sum((w**(2 * j + 1) / F(2 * j + 1) for j in range(12)), F(0))
    return lo, lo + 2 * w**25 / (25 * (1 - w * w))


ln3_lo, _ = log_bounds(F(1, 2))
_, ln2_hi = log_bounds(F(1, 3))
assert (ln3_lo - 1) / (2 * ln2_hi) > F(16, 225)
assert F(2449, 1000)**2 < 6
assert F(16, 225) * (6 + F(2449, 1000)) / 12 > F(1, 20)
print('[PASS] GO-4 k > 16/225 by rational log bounds; k (6+sqrt(6))/12 > (16/225)(6+2449/1000)/12 > 1/20',
      flush=True)
print('All GO checks passed.')
