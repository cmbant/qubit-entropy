"""PK5-4: exact support and spanning checks for the twelve-face candidate.

The dyadic vectors define a positive baseline, not a sharp dual identity.
The coefficient residual and repair bounds are checked separately by FP5-1.
"""
import json
import hashlib
from pathlib import Path

import flint
from sympy import isprime
from five_copy_candidate_data import load_candidate

DATA = Path(__file__).resolve().parent.parent/'data'
P = 2147483629
assert isprime(P)


def main():
    original = json.loads((DATA/'five_copy_ico_physical_faces.json').read_text())['faces']
    allv = json.loads((DATA/'five_copy_ico_allv_physical_faces.json').read_text())['faces']
    others = json.loads((DATA/'five_copy_ico_unflipped_oneflip_physical_faces.json').read_text())['faces']
    physical = {(2, q['tag']): q for q in original+allv}
    physical.update({(q['flip'], q['tag']): q for q in others})
    candidate = load_candidate()
    digest = hashlib.sha256()
    kernel_digest = hashlib.sha256()
    for rec in candidate:
        for row in rec['kernel_columns']:
            for value in row:
                kernel_digest.update(f"{value}\n".encode())
        for row in rec['basis_numerator']:
            for value in row:
                digest.update(f"{value}/2^{rec['basis_denominator_exponent']}\n".encode())
    assert digest.hexdigest() == '83fb29a61d84657ce1a4bf0c0c1c85799a72972bfd26587ca9ff8d73ed0a16d9'
    assert kernel_digest.hexdigest() == '0ad2b13be7b8f0cd8802be0d8aeb1975a408ecef4f860e8c875daede436086cd'
    print('PK5-4 triangular dyadic baseline fingerprint checked exactly; physical kernel columns unchanged')
    assert len(candidate) == 12
    sizes = [len(rec['basis_numerator']) for rec in candidate]
    assert sum(k*k for k in sizes) == 48434
    assert sum(k*(k+1)//2 for k in sizes) == 24502
    print('PK5-4 triangular factor slots 24502; dense factor slots 48434')
    used = set()
    for rec in candidate:
        flip = rec['flip']
        tag = rec['tag'] if flip == 2 else rec['tag'].split('-', 1)[1]
        key = (flip, tag)
        assert key not in used
        used.add(key)
        face = physical[key]
        n, rank = face['size'], face['rank']
        K = flint.fmpz_mat(rec['kernel_columns'])
        C = flint.fmpz_mat([[int(v) for v in row] for row in face['physical_columns']])
        assert (K.nrows(), K.ncols()) == (n, n-rank)
        assert K.transpose()*C == flint.fmpz_mat(n-rank, rank)
        assert flint.nmod_mat(n, n-rank, [int(v) % P for v in K.entries()], P).rank() == n-rank
        B = flint.fmpz_mat([[int(v) for v in row] for row in rec['basis_numerator']])
        assert B.nrows() == B.ncols() == n-rank
        assert all(B[i, j] == 0 for i in range(n-rank) for j in range(i+1, n-rank))
        assert all(B[i, i] > 0 for i in range(n-rank))
        assert 0 <= rec['basis_denominator_exponent'] <= 160
        assert any(int(v) % 2 for v in B.entries()), 'cancel common powers of two'
        assert rec['coefficient'] == int(rec['coefficient']) and rec['coefficient'] > 0
        print(f'PK5-4 {rec["tag"]}: exact physical support {n-rank}/{n}; dyadic basis invertible')
    assert used == set(physical)
    print('PK5-4 baseline file bytes', sum((DATA/name).stat().st_size for name in
          ['five_copy_twelve_face_candidate_basis.json', 'five_copy_baseline_dyadics.npz']))
    print('PK5-4 PASS: twelve full physical kernels; identity Grams give a strict positive baseline')
    print('Sharp coefficient identity and verified residual repair are separate FP5-1 checks')


if __name__ == '__main__':
    main()
