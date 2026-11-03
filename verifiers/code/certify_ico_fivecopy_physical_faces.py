"""PK5-1: exact small-integer physical kernels for all four repair faces.

Uses the complete rational Young invariant models and the exact common
feature ranks already certified by IK5-1--4.  Reconstructs independent
ambient feature columns and applies the physical Gram before checking the
new annihilators.  It does not supply any dual coefficient identity or PSD
multiplier.  --write creates the compact candidate artifact before checking.
"""
import argparse
import json
from pathlib import Path
import time

import flint
import numpy as np

from five_copy_physical_faces import (models, independent_columns,
                                      physical_columns, short_kernel)

DATA = Path(__file__).resolve().parent.parent/'data'
PAIRS = {'7': DATA/'five_copy_ico_g34_pair7.npz',
         '6': DATA/'five_copy_ico_g34_pair6.npz',
         '51': DATA/'five_copy_ico_g34_pair51.npz',
         '42': DATA/'five_copy_ico_g34_pair42.npz'}
ARTIFACT = DATA/'five_copy_ico_physical_faces.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    records = [] if args.write else json.loads(ARTIFACT.read_text())['faces']
    start = time.perf_counter()
    for index, (tag, U, factors, denominator, rank) in enumerate(models()):
        pair = np.load(PAIRS[tag])
        A = np.hstack([pair['A'], pair['B']])
        if args.write:
            columns = independent_columns(A, rank)
        else:
            record = records[index]
            assert record['tag'] == tag and record['rank'] == rank
            assert record['size'] == U.shape[1] and record['gram_denominator'] == denominator
            columns = record['source_columns']
        assert len(columns) == rank and len(set(columns)) == rank
        Q = A[:, columns]
        # A lower rank bound for the chosen columns; the IK certificates
        # provide the matching upper bound on the whole common feature image.
        Qm = flint.fmpz_mat(Q.shape[0], Q.shape[1], [int(v) for v in Q.ravel()])
        assert Qm.rank() == rank
        C = physical_columns(U, factors, Q)
        assert C.rank() == rank
        if args.write:
            K = short_kernel(C, rank)
            records.append({'tag': tag, 'size': U.shape[1], 'rank': rank,
                            'gram_denominator': denominator, 'source_columns': columns,
                            'physical_columns': [[str(C[i, j]) for j in range(rank)]
                                                 for i in range(C.nrows())],
                            'physical_kernel_rows': [[int(K[i, j]) for j in range(K.ncols())]
                                                      for i in range(K.nrows())]})
        else:
            expected = flint.fmpz_mat([[int(v) for v in row]
                                       for row in record['physical_columns']])
            assert C == expected
            K = flint.fmpz_mat(record['physical_kernel_rows'])
        assert K.nrows() == U.shape[1]-rank and K.ncols() == U.shape[1]
        assert K.rank() == U.shape[1]-rank
        assert K*C == flint.fmpz_mat(K.nrows(), rank)
        largest = max(abs(int(v)) for v in K.entries())
        print(f'PK5-1 {tag}: exact physical rank {rank}/{U.shape[1]}, '
              f'integer kernel {K.nrows()}x{K.ncols()}, max coefficient {largest}; '
              f'elapsed={time.perf_counter()-start:.2f}s', flush=True)
    if args.write:
        ARTIFACT.write_text(json.dumps({'class': 'exact physical kernels; no dual supplied',
                                       'faces': records}, separators=(',', ':'))+'\n')
    print('PK5-1 PASS: four physical kernels checked; dual equality and PSD remain OPEN')


if __name__ == '__main__':
    main()
