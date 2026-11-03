# Four- and seven-qubit entanglement entropy: companion material

**Compiled PDFs:** [Main paper](paper/four_qubit_entropy.pdf) · [Supplement](paper/supplement.pdf)

This is the companion to the paper *The most entangled states of four and seven qubits*.
The public repository is [cmbant/qubit-entropy](https://github.com/cmbant/qubit-entropy). The paper and supplement
give the proofs; this repository also holds the certificates and formalisation needed to check them:

| folder | contents |
|---|---|
| `paper/` | `four_qubit_entropy.pdf` (the paper) and `supplement.pdf` (its supplement: the full proofs of the lemmas the paper states with a proof idea, the certificate tables and further results, numbered S1, S2, …; a number without S refers to the paper), with the TeX sources of both; the paper's seven-qubit results section is `paper/seven_qubit_results.tex`, with the copy hierarchy and the ingredients of both seven-qubit proofs in `paper/seven_qubit_appendix.tex`; `paper/seven_qubit/` holds the seven-qubit parts of both documents: the paper's icosahedral state, and the supplement's parts (replica hierarchy, the four-copy construction `four_copy_bound.tex`, sextic covariants, purity--time-reversal tradeoffs, exact balanced-purity minima for `n ≤ 18`, the five-copy proof of the purity minimum `121/917` and the proof of the seven-qubit entropy maximum) |
| `verifiers/` | the exact, symbolic and interval-arithmetic verifiers cited in the paper's verification appendices (`code/`), certificate data (`data/`), their captured output (`verification/`), and `MAP.md` |
| `lean/` | a standalone Lean 4 / Mathlib formalisation of the four-qubit entropy maximum, its equality case and most of the four-qubit results, with parts of the seven-qubit and general-`n` results (listed in the Lean section) |

The tags in the paper (for example `[exact certificate; GO-1]`) name the rows of its verification appendices
(the exact finite checks appendix of the paper and Section S20 of the supplement). A theorem's checks are listed there, under the theorem's name.
`verifiers/MAP.md` lists, for each tag, the script in `verifiers/code/` that backs it; its captured output is
`verifiers/verification/<script>.txt`. The paper and supplement share the tags.

## Where each result is checked

The *paper* and *supplement* columns give each result's number and page in the two PDFs; a dash means the result is not in that PDF. For a result the paper states
and the supplement proves in full or extends, the supplement column gives the section that holds the proof or the further results. The checks are the tags of the verification sections (the exact finite checks appendix of
the paper and Section S20 of the supplement) and of [`verifiers/MAP.md`](verifiers/MAP.md), which gives the script; its captured
output is `verifiers/verification/<script>.txt`. A dash in the checks column means the proof is analytic and has no finite
check. The Lean names are explained in the Lean section below; † marks a theorem built only by the quadratic-variance target and
‡ one built only by the HS overlap target (both described there, with compiled certificate checks). Module and namespace names
beginning `PaperE` refer to this paper. The tags BF, MS, RG, HI, SF, SDS, NG, SW, E13L, TA and the `n = 13, 15, 19` rows of LP, GS
and QP back further results of the supplement that have no row here; `verifiers/MAP.md` and the supplement's verification
section list them under their results.

| result | paper | supplement | checks (tags) | Lean |
|---|---|---|---|---|
| Maximum and equality | Thm 1, p. 5 | – | ME-1–ME-6, SP-1–SP-5, ER-1–ER-5, CN-1–CN-4 | `meanEntropy_eq_max_iff` |
| Purity identity | Lem 2, p. 6 | – | ME-1–ME-6 | – |
| Moment inequality | Thm 3, p. 7 | – | ME-1–ME-6, SP-1–SP-5, SI-1–SI-3 | `sexticF_ge`, `sexticF_ge_sharp` |
| Complete moment zero set | Prop 14, p. 26 | Sec S5, p. 8 | ER-4–ER-5, PZ-1–PZ-2 | `moment_zeros` |
| One-cut rigidity | Prop 4, p. 9 | – | ER-1–ER-5 | `one_cut_rigidity_12/13/14` |
| Cubic majorants, Tsallis entropies, Rényi entropies below one | Prop 6, p. 10<br>Cor 7, p. 10<br>Cor 8, p. 11 | – | EF-1–EF-7 | `majorant_le`, `majorant_eq_iff_of_pos`, `tsallis_le`, `renyi_le` |
| Rényi envelope | Cor 9, p. 11 | Sec S6, p. 9 | RA, SCB, OT, UR, LT, HD, CG, SHD, LHD, NU, SL, BC, DA, CV, CJ, QVA, QV, QG, SD, JI | `renyi_rank_two_*`, `renyi_two_up`, `renyi_crossing`, `single_crossing_statement`, `meanRenyi_order_transfer`, `renyi_envelope_above_one_unconditional`† |
| General and critical envelopes, spin-flip derivative | Thm 15, p. 26<br>Prop 13, p. 25 | Lem S1, p. 4<br>Sec S3, p. 3<br>Sec S4, p. 6 | CE-1–CE-5, ESH-1–ESH-3, FG-1–FG-9, FI-1–FI-5 | `general_envelope_homog`, `general_envelope_coeff_iff`, `envelope_equality_iff`, `critical_closed_form`, `critical_envelope_eq_iff`, `flipGradient` |
| Convex roofs | Cor 10, p. 11 | – | – | `convexRoof_*` |
| Sharp quadratic purity variance (QV) and (QK) | Thm 27, p. 38<br>Cor 28, p. 38 | Sec S6.1, p. 10 | QVA-1–QVA-3, QV-1–QV-3, QG-1, QG-3, SD-4 | `sharpQuadraticVarianceBound`†, `quadraticVarianceBound`†, `qk_equality_classification_product`†, `qk_coefficients_sharp`† |
| Sharp HS overlap stability `Q ≥ 1−Φ` | Thm 18, p. 30 | Sec S7.2, p. 39 | GO-1–GO-3, NSL-1–NSL-3 | `globalOverlap_all`‡ |
| Global orbit stability | Thm 19, p. 30 | Sec S7, p. 38 | GO-4, OS-1–OS-3, EH-1–EH-3 | `orbitDist_sq`, `fidelity_floor`, `fidelity_floor_ghz_attained`, `entropy_gap_orbit_stability_dist_all`‡ (local-radius clauses not formalised) |
| Sharp critical stability | Thm 20, p. 31 | Sec S8.2, p. 47 | CO-1–CO-3, NC-1–NC-4, FP-1–FP-4, RP-1–RP-4 | `critical_orbit_sharp_dist_all`‡, `null_critical_sharp` |
| Sharp entropy stability on the critical locus | Thm 21, p. 31 | Sec S8.5, p. 58 | PM, PS, TC-1–TC-8, EX, LCC, CCV (covering Prop S40); numerical evidence NV-1, NV-2 | – |
| Critical normal form | Thm 16, p. 27 | – | CN-1–CN-4 | `critical_normal_form`, `critical_iff_LU_Gv` |
| Seven qubits: purity minimum `121/917` and unique icosahedral orbit | Thm 11, p. 13 | Sec S11, p. 71<br>Cor S67, p. 83<br>Thm S69, p. 84<br>Thm S71, p. 86 | S5-1, Y5-1, IK5-1–IK5-4, PK5-1–PK5-4, DP5-1, FP5-1–FP5-4, TT-1, MO-1, MO-2, FO-1, CC-1, CP-4, LH-1–LH-4, QB-1–QB-3 | abstract steps only: `SevenFiveCopy.tangent_line` (`lem:fivecopy-tangent-line`), `equality_moment_centred` |
| Seven qubits: icosahedral state | Thm 41, p. 46 | – | IC-1–IC-6 | `icosahedralState_eq_ico`, `pairCut_spectrum` (family spectra and purities) |
| Seven qubits: mean-entropy maximum `E₃(ψ_{x_*})` | Thm 12, p. 13 | Sec S12, p. 87<br>Thm S86, p. 99<br>Thm S88, p. 100<br>Thm S89, p. 101 | ES, SEH, EB, SC-2, SP7, AH, E7-K0–E7-K3, E7-1, FO-1, EO-1 (supplement), EG-1 | `star_value` (family value), `joint_scalar_majorant`, `joint_scalar_majorant_eq_iff` (AH-6), `seven_entropy_max_reduction_joint` (reduction assuming `π₃ ≥ 121/917` and E7-1) |
| Replica hierarchy: three- and four-copy levels (supplement) | – | Thm S45, p. 64<br>Thm S47, p. 66<br>Prop S49, p. 67 | TCB, FC-0–FC-4, NS-0–NS-4 (numerical TCN, FS) | – |
| Balanced-purity minima for `n ≤ 18`, `n ≠ 7, 13` (supplement) | – | Thm S101, p. 109<br>Thm S103, p. 110<br>Thm S104, p. 110<br>Thm S105, p. 111 | LP, GS, GL, E18-1–E18-4, EU-1–EU-2 | `minPurity<n>`, `shadow_nonneg`, `purity_gsn`, `entropyCeiling<n>` |
| Purity–time-reversal tradeoffs, even and odd `n` (supplement) | – | Thm S108, p. 112<br>Thm S112, p. 113 | ST-1–ST-9, TY-1–TY-2 | odd `n`: `odd_tradeoff_sharp`, `odd_tradeoff_sharp_eq_iff`, `ghz_odd_tradeoff_sharp_eq`; `n=4`: `purity_tradeoff`; `n ∈ {8,10,14,16,20}`: `purity_spinflip_tradeoff_even`, `_sharp`, `_coeff_optimal`, `_stability` |
| Seven-qubit stabilizer purity `19/140` | Prop 43, p. 48 | – | GS-4 | `pi3_witness` (the witness value only) |

To check one result, run the scripts of every tag in its row: a row lists the checks its proof uses, inputs included. For
example, the sharp variance bound QV needs the certificate QVA-1–QVA-3 together with the determining set and invariant-space
checks QG-1, QG-3 and SD-4; the critical constant `C_*` needs the exact contacts EX, the local cone LCC and the full covering
CCV; the seven-qubit purity minimum needs the physical kernels IK5/PK5 with the repair certificate FP5-1, and its equality case
MO-1, MO-2 and FO-1. The seven-qubit entropy maximum also uses the purity minimum, so its checks are those of both rows.

The Rényi proof above order one uses two anchors, `q_s = 1587/1000` and `q_u = 8/5`,
on complementary regions of the cut moments. The analytic order-transfer inequality (OT)
and monotonicity in the order extend those bounds to every `q > 1`; the paper's coverage
table gives the ranges for each region. The upper anchor is analytic (HD checks its algebra).
The unrestricted joint quadratic purity-variance inequality QK remains a certificate input: it is a corollary of the sharp bound QV, proved by the six-copy certificate QVA (analytic on critical states).
The lower anchor is analytic except for exact polynomial sign checks: the binary quartic lemma (monotonicity of A_q, by Cauchy's formula) needs none, SHD
adds eight one-variable quartic sign checks (SHD-3), and LHD (two Hermite bounds on the large-head chart) rests on the Bernstein
checks LHD-1, LHD-2 and LHD-4; LT reduces the remaining charts.
NU and SL handle the cluster neighbourhood, where a strict uniform dual slack is impossible.
In Lean the envelope for every `q > 1` is proved unconditionally (`renyi_envelope_above_one_unconditional`, from the formalised
QV and the corollary step `quadraticVarianceBound_of_sharp`); the dual lemma,
determinant monotonicity, low-tilt reduction, HS supporting inequality, bounded-head bound SHD, large-head curvature and bound LHD,
binary quartic lemma at `q_s` (`D_q'''>0` and monotone `A_q`, by a real-variable route), capped cubic bound CG, NU, cluster sliver SL, large-slack regime and
case analysis are proved outright.

## Verifiers

You need Python 3.11+ with `numpy`, `scipy`, `sympy`, `mpmath`, `python-flint`, `numba` and `opt_einsum` (`numba` and `opt_einsum` for the five-copy evaluators of the seven-qubit certificates; `cvxpy` for the `explore_*` SDP scripts and `four_copy_relaxation.py`, `matplotlib` for `make_figures.py`). `verifiers/requirements.txt` pins the versions
the logs were captured with (Python 3.13): `pip install -r verifiers/requirements.txt`. Then run a verifier
from `verifiers/code/`:

```bash
cd verifiers/code
python3 verify_m4_entropy_maximum.py
```

Several scripts read tables from `paper/*.tex`, so keep `verifiers/` and `paper/` side by side as shipped. The data files are
in `verifiers/data/` and `verifiers/code/data/`; the largest are the seven-qubit entropy bases (`verifiers/data/seven_entropy/`, about
2.9 MB) and the HS overlap certificate `verifiers/code/data/m4_global_overlap.json` (4.3 MB; its checker runs in under a minute).
Compare a script's output with `verifiers/verification/<script>.txt`. Exact and symbolic checks print the same values;
run-specific receipts may differ: timing strings, the box and attempt counts of the interval coverings (which depend on the
float heuristic that orders the boxes, hence on the `numpy` version), and the floating bounds printed by the repair
certificates (FP5-1, E7-1). What matters is that every `[PASS]` line and its stated pass condition (a margin, a bound below
its threshold, a covering with no undecided box) holds.

- **Naming.** `verify_*` scripts are fast checks. `certify_*` scripts are the longer interval coverings and certificates;
  most take 3 to 12 minutes on four cores. FP5-1 uses triangular dyadic inputs (441,612 bytes), thin-column repair actions and single-column norm bounds; its log records a complete cold run. E7-1 uses compact invariant-coordinate factors and exact image-column tables (about 2.9 MB), seven-column symmetric repairs and norm companions including Arb input radii; its recorded cold run took about 34.5 minutes. The slowest `verify_*` scripts take several minutes. `explore_*` scripts give numerical evidence only and back no proof.
- **Sign checks.** Interval and ball arithmetic use `python-flint` (Arb). Exact arithmetic uses `fractions`
  and `sympy`. No unbounded floating-point sign is used: the two seven-qubit repair certificates (FP5-1, E7-1) evaluate in
  binary64 and carry rigorous forward-error bounds for that rounding.

## Lean

```bash
cd lean
lake exe cache get   # downloads the Mathlib build cache (network, about a minute)
lake build
```

The default targets are the development and the general library. The development is `lean/FourQubitEntropy/`; `lean/FormalResearch/NQubit/` holds the general `n`-qubit
library (vectors, local `SU(2)` rotations, time reversal, Pauli weights, reduced states, entropy–purity bounds, Rains shadow inequalities, graph states), and `lean/QVCheckImpl/` the compiled quadratic-variance checker. The toolchain is pinned in
`lean-toolchain` and Mathlib in `lake-manifest.json`. With the cache, the default parallel build
takes about an hour on four cores and 15 GB of memory (measured); the heaviest files need about 9 GB each. Lake compiles
one module per core, so on a machine with 16 GB or less build the two heaviest modules first, one at a time:

```bash
cd lean
lake build FourQubitEntropy.PurityLPGraphData18 && lake build FourQubitEntropy.SpinFlipTradeoffEvenGraph20 && lake build
```

The HS overlap stability Theorem 18 also needs its certificate data, generated (byte-reproducibly, about 22 MB)
from `verifiers/code/data/m4_global_overlap.json`; the modules that use it are not in the default build:

```bash
cd lean
python3 FourQubitEntropy/gen_go_cert.py          # needs sympy; writes FourQubitEntropy/GOData/
lake build FourQubitEntropy.GOGlue.Final         # about 9 minutes, 7.5 GB
```

The sharp quadratic variance bound needs no generation step: its certificate data are committed Lean source
(`FourQubitEntropy/QVCert/Data.lean`, the certificate `verifiers/code/qv_all_state_certificate.json` as Lean literals, and
the checker's auxiliary tables `QVCheckImpl/DataStr.lean`, which the checker re-verifies). Its certificate check runs as
compiled code (about 14 minutes), so it is also outside the default build. The target below builds the bound
(`FourQubitEntropy/QVCheck/Unconditional.lean`) and its equality and optimality statements (`QVEqualityMain.lean`: `qk_equality_classification`,
`qk_equality_classification_product`, `qk_coefficients_sharp`, `qv_coefficient_iff`):

```bash
cd lean
lake build FourQubitEntropy.QVEqualityMain
```

**What is formalised.** There is no `sorry`. Each main theorem's docstring gives the paper's TeX label; `grep` it in `paper/*.tex`.
- **Moments.** The sharp sextic inequality `F ≥ (3/4) N W₁ ≥ 0`, with `N = ‖ψ‖²`, for all `ψ ∈ (ℂ²)^{⊗4}` (`sexticF_ge`), from the
  identity (S), and the optimality of `3/4` (`sexticF_ge_sharp`). The purity
  bound `P ≥ N²`. The optimal slope of the moment line (`moment_line_slope_optimal`).
- **Entropy maximum.** The bound `1 + ½ log₂ 3` on the mean balanced-cut von Neumann entropy, attained by `M₄`.
  Its equality case `meanEntropy_eq_max_iff`: equality holds iff `ψ` is a local-unitary image of `M₄` or of `conj M₄`.
  One-cut rigidity on all three cuts (`one_cut_rigidity_12/13/14`).
- **Spectral functionals.** The cubic-majorant criterion with its gap identity (`majorant_gap`, `majorant_le`,
  `majorant_eq_iff`). The Tsallis bound for `0<q<2` with its equality case (`tsallis_le`, `tsallis_eq_iff`). The
  Rényi bound for `0<q<1` (`renyi_le`, `renyi_eq_iff`). Sharpness of the Tsallis range: for `q>2` the cluster
  state beats `M₄` (`tsallis_cluster4_gt`).
- **Quadratic deficit bounds.** Quadratic lower bounds on the entropy and Tsallis deficits in terms of `W₁` and `τ`
  (`entropy_stability`, `tsallis_stability`, `tsallis_two_gap`). The paper's sharper estimates, which add `(a/12)(τ − τ^{3/2})` through Theorem 15, are `majorant_stability_env`, `entropy_stability_env` and `tsallis_stability_env`.
- **Envelopes.** The spin-flip derivative identity (`flipGradDefect_zero`, `flipGradient`; a 32-variable polynomial identity checked by the kernel in 22 declarations); the general envelope (`general_envelope_unit`) with its equality classification (`envelope_equality_iff`: product states or the critical equality family), with its coefficient `7/8` optimal (`general_envelope_coeff_iff`) and the critical envelope Proposition 13 in full: the bound (`critical_envelope`), its closed form (`critical_closed_form`), its equality classification (`critical_envelope_eq_iff`) and the critical moment boundary (`critical_moment_envelope_attained`), with a family of states attaining it (`critical_envelope_attained`).
- **Rényi structure.**
  - The rank-two envelope on every cut, with its equality case and the attaining cluster-to-Bell family
    (`renyi_rank_two_12/13/14`, `renyi_rank_two_sharp`).
  - The `q ≥ 2` bound `E_q ≤ 5/3 − ⅓ log₂(2P−1)` with its equality cases (`renyi_two_up`, `renyi_two_up_eq_gt`).
  - The HS/cluster crossing: a unique `q_*` with `1587/1000 < q_* < 1589/1000` (`renyi_crossing`, `crossing_fine`).
  - The single crossing Lemma 25 for every `q > 1`, with its equality case (`single_crossing_statement`),
    and the order transfer Corollary 26: a bound `E_{q₁} ≤ h_{q₁}` passes to every `1 < q < q₁`, with
    equality only for HS spectra (`renyiG_monotoneOn_all`, `renyi_order_transfer_all`, `meanRenyi_order_transfer`), and the
    anchor-order bounds of Theorem 36 (`classicalRelaxation`, in `FourQubitEntropy/QVCheck/Unconditional.lean`).
  - The sharp third-moment lemma with its equality classes (`third_moment`).
- **Convex roofs.** The mixed-state ceiling for the von Neumann, Tsallis (`0<q<2`) and Rényi (`0<q<1`) convex roofs
  (`convexRoof_vn_le`, `convexRoof_tsallis_le`, `convexRoof_renyi_le`), and its equality case: the mean convex
  roof equals the ceiling iff `ρ` is a pure state in the `M₄` or `conj M₄` orbit (`convexRoof_vn_eq_iff`,
  `convexRoof_tsallis_eq_iff`, `convexRoof_renyi_eq_iff`).
- **Chirality.** No local operator, even non-unitary, maps `M₄` to `conj M₄` (`no_local_operator_M4_conj`,
  `not_LUEquiv_M4_conj`).
- **Critical states** (namespace `PaperECritical`). The critical normal form Theorem 16
  (`critical_normal_form`, `critical_iff_LU_Gv`, with Lemma B.2 as `blochSq_eq_zero_iff` and the square case
  of Lemma 17 as `real_gram_square_SO`, the rectangular lemma as `real_gram_rect`); the zero set of `F`, Proposition 14: for unit `ψ`, `F(ψ) = 0`
  iff `ψ` is LU-equivalent to `M₄`, `conj M₄` or a product of two Bell pairs (`moment_zeros`); the inequalities of the
  Rényi grouping lemma and the cubic bound (`renyi_grouping`, `renyi_cubic_bound`), with their equality cases
  (`renyi_grouping_eq_iff`, `renyi_cubic_eq_iff`) and the necessity of `q_b` (`renyi_cubic_iff`); two-flat rigidity
  Proposition 40 (`two_flat_rigidity`, `two_flat_rank`; the first inequality of (TF) with its `W₁/4` term,
  `two_flat_cross_cost_12`; both inequalities of (TF), their sharpness and the critical-locus equality case,
  `two_flat_TF_12/13/14`, `two_flat_eq_iff_12/13/14`), the (QF) identification of states with two maximally mixed cuts with their Rényi values
  (`two_flat_QF`, `two_flat_renyi`); the spectral spin-flip bound Proposition 38 (`spectral_spinflip_12/13/14`,
  `spectral_spinflip_coeff_optimal`) from the isospectral overlap lemma Lemma 39
  (`isospectral_overlap` for square coefficient matrices; `isospectral_overlap_rect` in general, with attainment
  `isospectral_overlap_attained`).
- **Stability.** `16/225 · Q ≤ E(M₄) − E(ψ)` (`entropy_gap_ge_Q`, `entropy_gap_ge_defect`; on states with `W₁ = 0`, `δ ≥ k (Q + 5τ/2)`, `entropy_gap_critical`). Sharp critical-orbit stability (`critical_orbit_sharp`): with the orbit fidelity `Φ` and `d² = 2 − 2√Φ` defined, a unit critical state has `Q ≥ max(1−Φ, 3/2−2Φ)` and `Q ≥ ((2+√2)/4) d²`, with both entropy bounds and sharpness at the Bell-pair products (`bell_product_sharp`); the case `Φ ≥ 1/2` uses `Q ≥ 1−Φ` (`critical_orbit_sharp_dist_all`). **HS overlap stability** Theorem 18: `Q ≥ 1 − Φ` for every unit state (`globalOverlap_all`), with the global distance estimates of Theorem 19 and Corollary 22 unconditional (`entropy_gap_orbit_stability_dist_all`, `orbitDist_sq_le_Q_all`, `majorant_gap_orbit_all`, `tsallis_gap_orbit_all`). The reduction to the closest-point normal slice and all its constraints is in `PaperEGoSlice*.lean`; the certificate checker and its soundness in `GOProbe/`, and the link between them in `GOGlue/`. Also the implication
  (QK) ⇒ (JV) of Theorem S25 (`joint_purity_of_QK`), unconditional with the formalised QK (`joint_purity_unconditional`).

- **Purity–time-reversal tradeoff at `n = 4`** (Remark 6). Every unit state has
  `π₂ ≥ 1/3 + τ/6`, and `1/6` is the largest coefficient (`purity_tradeoff`, `purity_tradeoff_coeff_optimal`).
  The products of two Bell pairs on all three pairings are zeros of `F` (`bell2_F`, `bell13_F`, `bell14_F`).
- **Seven-qubit icosahedral state** (Theorem 41). `SevenQubitIcoState.lean` defines the coefficient tensor, checks
  the exact support counts and orthogonality, proves normalisation, and derives the maximally mixed one-qubit
  marginals and the reduced-matrix entries of one three-qubit cut of each type, directly from the tensor.
  `SevenQubitIcoPurity.lean` then computes the purity polynomial of the reduced state of every three-qubit cut
  (35) and every pair (21): `114α⁴+12α²β²+2β⁴` for three-qubit cuts avoiding the special site and
  `146α⁴+4α²β²+2β⁴` for those containing it, so the mean three-body purity is `121/917 + (1048/525)(x−37/524)²` on
  the family, with minimum `121/917` exactly at `x = 37/524`, and the cut and pair purities there are the values of
  Theorem 41. `SevenQubitIcoSpectrum.lean` (with the kernel certificates of `SevenQubitIcoCert.lean`) proves the
  spectrum of the reduced matrix of every three-qubit cut, `(4α²)⁶, (3α²+β²)²` avoiding the special site and
  `8α², (4α²)⁵, (α²+β²)²` containing it, and `SevenQubitIcoEntropy.lean` shows that the resulting mean three-body
  entropy of the family is strictly concave with a unique maximiser strictly between `0` and `37/524`
  (Corollary 42). `SevenQubitIcoPairSpectrum.lean` gives the spectrum of every pair cut,
  `2(8α²), 2(7α²+β²)` avoiding the special site and `2(10α²), 2(5α²+β²)` containing it (`pairCut_spectrum`), and
  `SevenQubitIcoMoments.lean` the family moments `π₂, π₃, b₃` with `b₃'(37/524) = −432/120127`. The finite checks use
  kernel `decide +kernel` and add no compiler axiom.
- **Seven-qubit maximiser** (Proposition S73, scalar part). The maximiser `x_*` of the family entropy
  satisfies `1/32 < x_* < 37/524` (`star_bounds`) and is the unique root in `(0,1/2)` of
  `2³²(1−2x)¹⁵ = 3⁸(1+8x)⁸(1+28x)⁷` (`star_equation`), with `0.0698287 < x_* < 0.0698288` (`star_enclosure`) and
  `2.9653560294776 < E₃(x_*) < 2.9653560294777` bits (`star_value`, exact rational log enclosures). The longer decimal
  of Theorem 12 is not formalised, and `E₃` here is the family function computed from the proved spectra.
- **Seven-qubit exact structure.** The coefficient tensor equals the two sums of (14) (`icosahedralState_eq_ico`);
  the joint spin-flip identity on all 35 cuts of the family (Lemma S80, `tripleCut_spinflip_family`); the
  family relations of Proposition S97 (`zeroFocus_one`, `zeroFocus_wronskian_unconditional`,
  `zeroFocus_three_ico_unconditional`, with `V_r = Σ_{|S|=r}(R_S⊗I)ψ` standing for `Φ_{∅,r}`).
- **Seven-qubit entropy maximum: inputs, and the reduction under explicit hypotheses.** Proved outright: the fixed-purity
  envelope (`entropy_le_g8`, the case `d = 8` of `entropy_le_gEnv`), the sharp linear-purity bound `3 − S ≥ (7 log₂7/48)(8 tr ρ² − 1) ≥ (49/120)(8 tr ρ² − 1)`
  (`entropy_linear_purity_sharp_bits`, `cutEntropy_linear_purity_sharp`; the weaker `2/5` form `entropy_linear_purity_bits`,
  `meanEntropy3_linear_purity`), the spectral pair Proposition S76 (`spinflip_spectral_pair`,
  `cross_cut`), the eigenvalue cap `497/1000` for pure states with `E₃ > (59307/20000) ln 2` in nats, strictly above the SC-2 threshold (`eigenCap_497_pure`; the paper's corollary also covers mixed states and the threshold itself), the minimal majorant degree `2k−1`
  (`majorant_degree_isLeast`; degree seven at the maximiser, `exists_maximiser_degree_seven`) and the SC-2 witness
  `(59307/20000) ln 2 < E₃(x_*)` (`sc2_witness`). Proved under the explicit hypothesis `SevenPurityBound`
  (`π₃ ≥ 121/917` for every state, a `Prop` argument, not an axiom): a Pauli-weight bound `15A₁+5A₂+A₃ ≥ 255/131` (`cor_sevenpauli`, an auxiliary consequence not stated in the paper) and the
  competitor band `121/917 ≤ π₃ < 8693/64000` (Proposition S79, `competitor_purity_band`). The final
  reduction of Theorem S86 (`seven_entropy_max_reduction`,
  `seven_entropy_max_reduction_of_scalarMajorant`, `seven_entropy_max_reduction_star`): from `SevenPurityBound`, a
  joint majorant bound (`JointMajorantBound`, or a scalar majorant `ScalarMajorant`) and the localised five-copy
  bound (`LocalizedBound`, E7-1), every state has `E₃ ≤ E_*`. The functional `𝒫₅` and the polynomial `(K,h)` are
  arbitrary, so the substantive Lean content is the contender-region step (`contender_region`) and the matrix part of the
  joint majorant (`meanEntropy3_le_jointP5`: any scalar majorant on `[0,497/1000]²` gives the joint bound on the contender
  region; `meanEntropy3_le_jointP5_unit`: any scalar majorant on `[0,1]²` gives it for every unit state). The joint scalar majorant AH-6 is formalised for the paper's coefficients
  (`joint_scalar_majorant`, one kernel check in integer ball arithmetic), so `seven_entropy_max_reduction_joint` needs only
  `SevenPurityBound` and E7-1 (`LocalizedBound`); those two and attainment of `E_*` are not formalised, so this is
  not a proof of the maximum.
- **Balanced-purity minima and entropy ceilings** (Table S6, Table 5). For every `4 ≤ n ≤ 18` except `7` and `13`,
  the minimum of the mean balanced purity `π_{⌊n/2⌋}` equals the table value (`minPurity<n>`): Rains' shadow inequalities
  (`shadow_nonneg`) and the linear-programme certificates give the lower bound, and graph states with purity `2^{-rk₂}`
  (`purity_gsn`) attain it. The entropy–purity slope `log₂d − S ≥ c_d(d tr ρ² − 1)`, `c_d = (d−1)log₂(d−1)/(d(d−2))`,
  holds for every `d ≥ 3` with both equality spectra (`entropy_purity_slope_bits`, `entropy_purity_slope_eq_iff`), and the
  fixed-purity envelope for every `d ≥ 2` (`entropy_le_gEnv`, `entropy_eq_gEnv_iff`); with the minima they give the
  entropy ceilings of Table 5 for `8 ≤ n ≤ 18`, `n ≠ 13` (`meanEntropy_le`, `entropyCeiling<n>`).
- **Odd-`n` purity–time-reversal tradeoff** (Theorem S112). For every odd `n ≥ 5`,
  `π_m + c_n O = 1/2 + Σ_{j odd} w_j A_j` with `c_5 = 1`, `c_n = (n+1)/4` for `n ≥ 7`, the closed-form weights, `w_{n−2} = w_n = 0`
  and `w_j > 0` for odd `j ≤ n−4` (`odd_tradeoff_sharp`, `sharpW_pred`, `sharpW_top`, `sharpW_pos`), the equality case
  (`odd_tradeoff_sharp_eq_iff`) attained by GHZ (`ghz_odd_tradeoff_sharp_eq`), and `O` as the mean T-odd purity
  (`oddPur_eq_half_sub_Rspin`).
- **Spin-flip doubling and the abstract five-copy equality steps.** Theorem S51 and Corollary S52
  (`spinflip_double_odd`, `ordinary_triple_purity`); the five-copy tangent-line lemma Lemma S63 in abstract
  form, for any copy number and any number of qubits (`SevenFiveCopy.tangent_line`; with the equality clause, `SpinFlip.tangent_line_eq`); and the moment-level step of
  Theorem S66 (`equality_moment_centred`: the tangent line and the elimination of `t`, given `H(X_y') ≠ 0`).
  These do not use FP5-1, which is not formalised.
- **Even-`n` purity–time-reversal tradeoff** (Theorem S108, Corollary S109). For
  `n ∈ {8, 10, 14, 16, 20}`, `π_{n/2} ≥ L_n + c_n τ_n` with the table's constants (`purity_spinflip_tradeoff_even`), attained by
  one all-odd-degree graph state per `n` (`purity_spinflip_tradeoff_even_sharp`, `tau_gsn_eq_one`), so `c_n` is optimal for
  `L_n` (`purity_spinflip_tradeoff_even_coeff_optimal`), and `π_{n/2} ≤ L_n + δ` gives `τ_n ≤ δ/c_n`
  (`purity_spinflip_tradeoff_even_stability`).

`FourQubitEntropy/Checks.lean` adds concrete evaluations on product, Bell-pair and cluster states.

**Trust boundary.** The main theorems, apart from those marked † and ‡ above, are checked by the Lean kernel alone: `#print axioms` lists only `propext`,
`Classical.choice` and `Quot.sound`. The default build also compiles the semantics of the quadratic-variance checker
(`FourQubitEntropy/QVCheck/`), which contains one `native_decide` fact (`cArr_rowC`) used only by that bound. Two results
use compiled evaluation, both outside the default build. The sharp quadratic variance bound (`FourQubitEntropy/QVCheck/Unconditional.lean`) proves the certificate
identity of Section S6.1 for every vector: the difference of its sides is a six-copy sesquilinear form that vanishes once its
copy-symmetrisation does, which reduces to one integer matrix identity per spin sector; those identities, the invariant
bases and the intertwiner table are checked by one compiled checker (`QVCheckImpl/`, a `precompileModules` library;
`checkAll_true`, `cArr_rowC`), and the soundness of the checker and every other step are kernel-checked. The other is the HS
overlap stability (`FourQubitEntropy/GOGlue/Final.lean`, outside the default build): three finite checks of the generated certificate data
(`check_all`: the 277 Gram blocks and the polynomial identity; `identCheck_true`: the target polynomial; `consCheck_true`:
the 64 constraint matrices) are evaluated by `native_decide`. Each `native_decide` step adds its own axiom, printed as `<declaration>._native.native_decide.ax_1_1`, so
`#print axioms` on these theorems lists those three (and on the quadratic-variance theorems, those of `checkAll_true` and `cArr_rowC`). The soundness of
these checks and everything else is kernel-checked. The 32-variable
sextic polynomial identity is split into seven pieces (`IdentityCert*.lean`), each a separate kernel-evaluated
declaration, and the spin-flip derivative identity likewise into 22 (in the 14 files `FlipGradientCert*.lean`). **Machine required:** these take 11–63 s each and up to about 9 GB of memory (including about 6 GB of
mapped Mathlib files), about a quarter of an hour in sequence. The packed linear-programme and graph-state data of the purity minima add about
another quarter of an hour. A 16 GB machine builds the whole project, building those two modules first as above.

**Not formalised.** The following results are outside the Lean development; their proofs and certificate inputs are in the paper and supplement:
- four qubits: the local-radius and limiting-coefficient clauses of Theorem 19 and Corollary 22
  (Lemma S28); Theorem 21 (`C_*`) with its covering proof (Proposition S40 and its lemmas);
  the critical fidelity profile (S8.8) (Lemma S30, Lemma S31);
  Theorem S32 for `θ > 0`; Proposition S29; the near-one proof of Theorem S18 (Lemma S19,
  the defect estimate `eq:fqe-renyi-defect` and the Rényi clause of Corollary 22) and Corollary S24 (the bound
  and equality case of Theorem S18 follow from `renyi_envelope_above_one_unconditional` and `renyi_crossing`);
  Proposition S3; the equality and defect clauses of Proposition S22 and the equality clause of Theorem S25;
  Lemma S26; Rényi convex roofs for `1 < q < q_*`;
- seven qubits: the global purity minimum `121/917` (Theorem 11, Corollary S67, FP5-1 and the five-copy
  chain) and entropy maximum (Theorem 12, Theorem S86: E7-1, E7-K0–K3), with the entropy certificates
  (Theorem S85, Proposition S87, the five-copy part of Proposition S84,
  Corollary S83, the mixed-state clause of Corollary S78), attainment by `ψ_{x_*}` and
  full-space stationarity (Lemma S72); orbit rigidity (Theorem S64,
  Corollary S65, Corollary S68; of Theorem S66 only the abstract moment-level
  step is formalised, `equality_moment_centred`: the tangent line and the elimination of `t`, given `H(X_y') ≠ 0`; the family value
  `H(ψ_y) = 155 + 96y` (FO-1) and the passage to the orbit are not);
  Theorem S69, Theorem S71, Theorem S88, Theorem S89, Lemma S70,
  Lemma S91, Corollary S90,
  Proposition S92; Proposition 43 (GS-4 checks its identities and the equality witness), Proposition S93,
  Corollary S94; Theorem S53, Proposition S54, Proposition S55;
- replica hierarchy and covariants: Proposition S43, Theorem S44, Theorem S45, Theorem S47,
  Corollary S46, Proposition S49, Proposition S95, Lemma S96, Proposition S97 beyond its family relations,
  Theorem S100, the weight distribution of Theorem S105 and the `n = 18` entry of
  Corollary S107.

## Rebuilding the PDFs

```bash
cd paper
latexmk -pdf four_qubit_entropy.tex; latexmk -pdf supplement.tex     # the paper and its supplement refer to each other,
latexmk -g -pdf four_qubit_entropy.tex; latexmk -g -pdf supplement.tex  # force the second pair for external references
```

`make_figures.py` in `verifiers/code/` regenerates `paper/figures/` (it imports `make_fig_cuts.py` and
`make_fig_icosahedron.py`).

## Licence

The code in `verifiers/` and `lean/` is MIT-licensed. The paper and its supplement in `paper/` are CC BY 4.0. See
`LICENSE`.

## Distribution

This tree is prepared for distribution as ancillary files of the arXiv submission as well as in the public repository.
