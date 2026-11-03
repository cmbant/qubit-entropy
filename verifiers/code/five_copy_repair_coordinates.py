"""Independent symmetric coordinates inside a thin Gram repair.

For an exact orthogonal Q and width r, a repair is
Q [[D, E], [E^T, 0]] Q^T, with D symmetric. Each upper-triangular D
entry and each E entry is used once. The repair's operator norm is at
most sqrt(2) times the Euclidean norm of these coordinates.
"""
from fractions import Fraction

import flint
import numpy as np
from five_copy_candidate_data import load_frames

UROUND = 2.0**-53
INFLATE = 1 + 2.0**-20


def exact_frames(path):
    out = []
    for rec in load_frames(path):
        k = len(rec["vectors"][0])
        identity = flint.fmpz_mat(k, k, [int(i == j) for i in range(k) for j in range(k)])
        numerator, denominator = identity, 1
        for values in rec["vectors"]:
            v = flint.fmpz_mat(k, 1, values)
            norm = sum(int(x)**2 for x in values)
            numerator = norm*numerator - 2*v*(v.transpose()*numerator)
            denominator *= norm
        assert numerator.transpose()*numerator == denominator**2*identity
        Q = np.array([[float(Fraction(int(numerator[i, j]), denominator))
                       for j in range(k)] for i in range(k)])
        nonzero = abs(Q[Q != 0])
        assert np.isfinite(Q).all() and nonzero.min() > 2.0**-100 and nonzero.max() < 2
        out.append((numerator, denominator, Q))
    return out


def frames(path):
    return [Q for numerator, denominator, Q in exact_frames(path)]


def transform(row, radius, Qs, cap=7):
    # A trusted older checkpoint may contain the original width-16 arrays.
    # New evaluations need only the first cap frame columns.
    old_sizes = [min(16, len(Q))*len(Q) for Q in Qs]
    new_sizes = [min(cap, len(Q))*len(Q) for Q in Qs]
    sizes = old_sizes if len(row) == sum(old_sizes) else new_sizes
    assert len(row) == sum(sizes)
    parts, extra, offset = [], 0.0, 0
    for Q, size in zip(Qs, sizes):
        k, r = len(Q), min(cap, len(Q))
        M = row[offset:offset+size].reshape(size//k, k)[:r]
        offset += size
        nonzero = abs(M[M != 0])
        assert np.isfinite(M).all()
        assert not len(nonzero) or (nonzero.min() > 2.0**-400 and nonzero.max() < 2.0**400)
        Y = M @ Q
        for a in range(r):
            for b in range(a, k):
                if a == b:
                    parts.append(Y[a, b]/2)
                elif b < r:
                    parts.append((Y[a, b] + Y[b, a])/2)
                else:
                    parts.append(Y[a, b])
        mn = np.linalg.norm(M)*INFLATE
        qn = np.linalg.norm(Q)*INFLATE
        gamma = k*UROUND/(1-k*UROUND)
        # Q is exactly orthogonal. Conversion costs at most u/(1-u)
        # times its computed Frobenius norm; the dot products have depth
        # at most k. Selection, halving and averaging contract error norms.
        extra += (gamma + UROUND/(1-UROUND))*mn*qn*INFLATE
        extra += UROUND*np.linalg.norm(Y)*INFLATE
    return np.array(parts), (radius + extra)*INFLATE
