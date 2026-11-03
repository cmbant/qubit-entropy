#!/usr/bin/env python3
"""QG-5: the raw QG Gram entries lie in the symmetric, conjugation-even LU-invariant space.

Proved in: quadratic_variance_certificate.tex (proof of thm:fqe-quadratic-variance).  The
determining-set argument needs every raw entry B_alpha(z) to be a symmetric,
conjugation-even local-unitary invariant of bidegree (6,6).  The construction gives this;
here it is checked exactly on a full-support Gaussian-integer state:
  * complex conjugation z -> conj(z) leaves every raw entry unchanged;
  * the three adjacent qubit transpositions (generating S_4) leave every entry unchanged;
  * three Clifford-type Gaussian-integer multiples of unitaries (u^dagger u = 2 I) and two
    non-Clifford ones (u^dagger u = 5 I, irrational rotation angles), applied on each of the
    four sites, multiply every entry by k^6 (bidegree (6,6), u^dagger u = k I).
The non-Clifford maps generate a dense subgroup together with the Clifford ones, so a
construction covariant only under the finite Clifford group would fail QG-5c.
Integer arithmetic only.  This is a regression of the construction, not a proof of it.
"""
import numpy as np
from pathlib import Path
import json
from quadratic_covariants import QuadraticCovariants

DATA = json.loads(Path(__file__).with_name('quadratic_purity_variance_certificate.json').read_text())
C = QuadraticCovariants(DATA)
rng = np.random.default_rng(20261023)
z0 = rng.integers(-3, 4, 16).astype(np.int64)
z1 = rng.integers(-3, 4, 16).astype(np.int64)
assert all(int(a) or int(b) for a, b in zip(z0, z1)), 'full support'
base, _ = C.features(z0, z1)
print(f'raw entries: {len(base)}; nonzero: {sum(1 for v in base if v)}')

conj, _ = C.features(z0, -z1)
assert conj == base
print('QG-5a PASS: complex conjugation fixes every raw entry')


def permute(v, axes):
    return v.reshape([2]*4).transpose(axes).reshape(16).copy()


for axes in ((1, 0, 2, 3), (0, 2, 1, 3), (0, 1, 3, 2)):
    perm, _ = C.features(permute(z0, axes), permute(z1, axes))
    assert perm == base, axes
print('QG-5b PASS: the adjacent qubit transpositions fix every raw entry')

# (A + iB) with A, B integer matrices; each satisfies u^dagger u = k I.
UNITARIES = {
    'hadamard-type': (2, np.array([[1, -1], [1, 1]]), np.zeros((2, 2), dtype=np.int64)),
    'x-rotation':    (2, np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])),
    'phase':         (2, np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, -1]])),
    'phase-1+2i':    (5, np.array([[1, 0], [0, 1]]), np.array([[2, 0], [0, -2]])),
    'rotation-2,1':  (5, np.array([[2, -1], [1, 2]]), np.zeros((2, 2), dtype=np.int64)),
}
# The k = 5 maps use a full-support state with entries in {-1, 0, 1}, keeping every
# integer contraction inside the int64 overflow bounds of integer_gram.
small = (np.array([1, -1, 0, 1, 1, 0, -1, 1, 0, 1, 1, -1, 1, 0, 1, 1], dtype=np.int64),
         np.array([0, 1, 1, -1, 0, 1, 1, 0, 1, -1, 0, 1, 1, 1, -1, 0], dtype=np.int64))
assert all(int(a) or int(b) for a, b in zip(*small)), 'full support'
small_base, _ = C.features(*small)
for name, (k, A, B) in UNITARIES.items():
    U = A + 1j*B
    assert np.array_equal(U.conj().T@U, k*np.eye(2))
    s0, s1, ref = (z0, z1, base) if k == 2 else (*small, small_base)
    for site in range(4):
        x, y = s0.reshape([2]*4), s1.reshape([2]*4)
        ax = lambda M, v: np.moveaxis(np.tensordot(M, v, axes=([1], [site])), 0, site)
        w0 = (ax(A, x) - ax(B, y)).reshape(16)
        w1 = (ax(A, y) + ax(B, x)).reshape(16)
        lu, _ = C.features(w0.astype(np.int64), w1.astype(np.int64))
        assert lu == [k**6*v for v in ref], (name, site)
print('QG-5c PASS: three Clifford-type and two non-Clifford local unitaries on each of the four sites scale every raw entry by k^6 (u^dagger u = kI)')
