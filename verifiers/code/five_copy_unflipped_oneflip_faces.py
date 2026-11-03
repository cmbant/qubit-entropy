"""Complete rational all-V models for the unflipped and one-flip faces."""
import itertools
import math
from fractions import Fraction
from functools import lru_cache

import numpy as np
import opt_einsum as oe

import verify_five_copy_site_algebra as s5
from five_copy_allv_young_models import invariant_basis
from five_copy_symmetric_models import invariant_bases
from five_copy_young51_model import monomials, sym_power, select_columns
from certify_ico_fivecopy_g34_51 import local_maps, spinflip


def mul(p, q):
    return tuple(p[q[i]] for i in range(5))


def action(U, k, l, R):
    A, B = sym_power(R, k), sym_power(R, l)
    assert len(A)*len(B)*int(abs(A).max())*int(abs(B).max())*int(abs(U).max()) < 2**63
    return np.einsum('ij,jbk,ab->iak', A, U.reshape(len(A), len(B), U.shape[1]),
                     B, optimize=True).reshape(U.shape)


def character_dimension(k, l, group):
    total = Fraction(0)
    for i in group:
        R = s5.RHO[2][i]
        Z = R**0
        traces = [0]
        for j in range(1, 8):
            Z = Z*R
            traces.append(int(Z.trace()))
        h = [Fraction(1)]
        for j in range(1, 8):
            h.append(sum(traces[a]*h[j-a] for a in range(1, j+1))/j)
        total += h[k] if not l else h[k]*h[l]-h[k+1]*h[l-1]
    return total/len(group)


@lru_cache(None)
def models():
    U7, _ = invariant_bases()
    base = [('7', U7, 7, 0), ('61', invariant_basis(6, 1), 6, 1),
            ('52', invariant_basis(5, 2), 5, 2)]
    H = [p for p in s5.PERMS if set(p[3:]) == {3, 4}]
    seen, reps = set(), []
    for p in s5.PERMS:
        if p in seen:
            continue
        reps.append(tuple(np.argsort(p)))
        seen.update(mul(h, p) for h in H)
    assert len(reps) == 10
    out = []
    for flip in (0, 1):
        if flip == 0:
            projection = reps
        else:
            projection = [tuple(range(5))]
            for i in range(3):
                p = list(range(5)); p[i], p[3] = p[3], p[i]
                projection.append(tuple(p))
        group = [i for i, p in enumerate(s5.PERMS) if not flip or p[4] == 4]
        for (tag, U, k, l), dim in zip(base, (4, 6, 9) if not flip else (16, 30, 39)):
            assert character_dimension(k, l, group) == dim
            Y = sum((action(U, k, l, np.array(s5.RHO[2][s5.PINDEX[p]], np.int64))
                     for p in projection), np.zeros_like(U))
            V = select_columns(Y)
            assert V.shape[1] == dim
            for i in group:
                assert np.array_equal(action(V, k, l, np.array(s5.RHO[2][i], np.int64)), V)
            out.append((flip, tag, k, l, V))
    return out


@lru_cache(None)
def contraction_path():
    # The path depends only on shapes. A single retained path is reused for
    # every integer input, avoiding input-dependent numerical decisions.
    state = np.zeros((2,)*7, np.int64)
    operands = tensor_operands(state, state, 0)
    path, _ = oe.contract_path(*operands, list(range(35, 42)),
                              optimize=oe.RandomGreedy(max_repeats=128))
    return path


def tensor_operands(z, jz, flip):
    operands = []
    for c in range(5):
        operands += [z if c < 5-flip else jz, [7*c+s for s in range(7)]]
    M = local_maps()[2].astype(float).reshape((10,)+(2,)*5).transpose((1,2,3,4,5,0))
    for s in range(7):
        operands += [M, [7*c+s for c in range(5)]+[35+s]]
    return operands


def physical_tensor(z, flip):
    assert flip in (0, 1)
    assert np.count_nonzero(z) <= 32 and int(abs(z).max()) <= 5
    # At most 32^5 state-support tuples; every local map entry has size <=3.
    assert 32**5 * 5**5 * 3**7 < 2**53
    operands = tensor_operands(z, spinflip(z), flip)
    T = oe.contract(*operands, list(range(35, 42)), optimize=contraction_path())
    assert np.array_equal(T, np.rint(T))
    return T.astype(np.int64).reshape(*([5,2]*7)).transpose(
        [2*i for i in range(7)]+[2*i+1 for i in range(7)])
