# Sharp seven-qubit entropy certificate inputs

The proof is `seven_qubit_entropy_certificate.tex`, part of the paper's supplement.
All stored floating entries are fixed exact dyadic rationals. Verification reads inputs, without a solver, git history or saved evaluations.

- `models.json`: seventeen ambient representations, copy subgroups and positive source weights.
- `bases.npz` (format 1): the original integer invariant bases `u0` through `u16`, stored in narrow
  integer types, and new lower-trapezoidal dyadic factors `b0` through `b16` in invariant coordinates.
  `seven_entropy_data.load_bases` promotes integer arrays to int64 before arithmetic.
  E7-1 projects the factors onto the exact physical kernels at `x_*` using Arb. Old-baseline equality is not assumed.
- `constraints.npz` (format 1): original degree-five physical images, with distinct full columns
  `c0` through `c16` and integer lookup arrays `i0` through `i16`. `load_constraints` expands every
  coefficient exactly to int64. Faces 1, 2 and 3 use Euclidean invariant coordinates; the others
  use physical Gram numerators. E7-K1 reconstructs the images; E7-K2 checks ranks and `pivots.json`.
- `cubic.json`: unchanged rational cubic bases, images, generating polynomials and local maps;
  `cubic_columns.json` and `cubic_baseline.npz` specify the kernel-projector columns and
  dyadic factors of the two positive cubic multipliers. E7-K3 checks the physical inputs.
- `joint_features.json` and `joint_choices.npz`: unchanged joint trace products and dyadic factors
  of the ordinary and cap-weighted marginal Grams. Their maps are reconstructed exactly.

The archives were generated once by least squares and QR, which only propose the dyadic baselines;
the full E7-1 checker establishes their validity independently, so no generator is needed to check them.
`verify_seven_entropy_compact.py` fingerprints all integer bases/images and compares the compact and
dense evaluators.

The parent data directory supplies the rational frames and all 3,424 probes with their
row weights. E7-1 uses 3,754 independent symmetric repair coordinates, validates rank 3,283 and
requires the inverse defect and Gram correction to be below one. The default is a cold run.
`--resume --work DIR` reuses only this user's trusted, source/input-bound atomic checkpoints;
checkpoints are generated scratch, never committed certificate inputs.
