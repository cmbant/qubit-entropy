# Verification tags and scripts

Each tag of the verification sections of the paper (its exact finite checks appendix) and of the supplement (its section of exact finite checks), with the
script (in `code/`) and data that back it. Each script's captured output is `verification/<script>.txt`.
Data files are in `data/`, `code/data/` and `code/`.

| tag | scripts and data |
|---|---|
| ME-1–ME-6 | `verify_m4_entropy_maximum.py` |
| SP-1–SP-5 | `verify_m4_entropy_swap_proof.py` |
| SI-1–SI-3 | `verify_m4_moment_sharp_w1.py` |
| CE-1–CE-5 | `verify_critical_sextic_envelope.py` |
| PZ-1–PZ-2 | `verify_polarization_remainder_zeros.py` |
| FI-1–FI-5 | `verify_flip_gradient_interpolation.py` |
| FG-6–FG-8 | `verify_envelope_flip_gradient.py` |
| ESH-1–ESH-3 | `verify_envelope_coefficient_sharp.py` |
| ER-1–ER-5 | `verify_m4_entropy_rigidity.py` |
| CN-1–CN-4 | `verify_critical_normal_form.py` |
| EF-1–EF-7 | `verify_m4_entropy_family.py` |
| RA-1–RA-5 | `verify_renyi_above_one.py` |
| SCB-1–SCB-5 | `verify_spectral_spinflip_bound.py` |
| UR-4 | `verify_renyi_uncapped_reduction.py` |
| OT-1–OT-4 | `verify_renyi_single_crossing.py` |
| SHD-1–SHD-4 | `verify_renyi_small_head.py` |
| LHD-1–LHD-4 | `verify_renyi_large_head.py` |
| BC-1–BC-3 | `verify_renyi_cubic_bound.py` |
| DA-1–DA-5 | `verify_renyi_dual_algebra.py` |
| CV-1–CV-5 | `verify_critical_purity_variance.py` |
| CJ-1–CJ-5 | `verify_critical_joint_variance.py` |
| QVA-1–QVA-4 | `qv_all_state_certificate.json`, `verify_qv_all_state_certificate.py` |
| EU-1, EU-2 | `verify_entropy_purity_upper_table.py` |
| QG-1–QG-5 | `quadratic_covariants.py`, `quadratic_purity_variance_certificate.json`, `spin_covariants.py`, `verify_quadratic_gram_invariance.py`, `verify_quadratic_variance_certificate.py` |
| RG-1–RG-3 | `verify_m4_correlation_geometry.py` |
| IC-1–IC-6 | `verify_icosahedral_seven_qubit_state.py` |
| EG-1 | `verify_seven_entropy_gap.py` |
| GS-4 | `verify_balanced_purity_lp_graph_states.py` |
| SD-1, SD-2, SD-4 | `verify_symmetric_invariant_dimensions.py` |
| GO-1–GO-4 | `m4_global_overlap.json`, `orbit_certificate_polynomials.py`, `verify_m4_global_overlap.py` |
| FG-1–FG-5, FG-9 | `verify_envelope_flip_gradient.py` |
| CO-1–CO-3 | `verify_m4_critical_orbit_overlap.py` |
| OS-1–OS-3 | `verify_m4_lu_orbit_stability.py` |
| BF-1–BF-4 | `verify_m4_bell_frame_stability.py` |
| EH-1–EH-3 | `verify_m4_entropy_orbit_hessian.py` |
| NSL-1–NSL-3 | `verify_m4_closest_slice.py` |
| NC-1–NC-4 | `verify_m4_null_critical_geometry.py` |
| RP-1–RP-4 | `verify_m4_critical_real_density.py` |
| FP-1–FP-4 | `m4_critical_covering_profile.json`, `verify_m4_critical_covering_profile.py` |
| PM-1, PM-2, PM-4, PS-1 | `verify_p1_cut_model.py`, `verify_p1_sharp_slice.py` |
| NV-1–NV-2 | `explore_critical_tau_profile.py`, `explore_noncritical_states.py` |
| TC-1–TC-8 | `verify_critical_chart.py` |
| EX | `verify_critical_apex_exact.py` |
| LCC | `critical_cone_lib.py`, `verify_critical_local_cone.py` |
| CCV | `certify_critical_covering.py`, `critical_cells.py`, `critical_cover_lib.py`, `critical_float.py`, `critical_side_jets.py` |
| UR-3 | `verify_renyi_uncapped_reduction.py` |
| HD-1–HD-5 | `verify_renyi_hs_analytic.py` |
| LT-1–LT-3 | `verify_renyi_low_tilt.py` |
| CG-1–CG-5 | `verify_renyi_capped_cubic.py` |
| NU-1–NU-3 | `moment_rectangle.py`, `verify_renyi_near_uniform.py` |
| SL-1–SL-4 | `verify_renyi_cluster_sliver.py` |
| QV-1–QV-3 | `verify_quadratic_purity_variance.py` |
| JI-1–JI-3 | `verify_joint_purity_implication.py` |
| MS-1–MS-4 | `verify_m4_null_mixed_geometry.py` |
| ES-1 | `verify_seven_entropy_stationarity.py` |
| SEH-1–SEH-3 | `certify_seven_qubit_entropy_hessian.py` |
| EB-0–EB-3 | `verify_seven_entropy_quant_basin.py` |
| SC-2 | `verify_seven_entropy_spectral_cap_improved.py` |
| SP7-1, SP7-5, SP7-6 | `verify_seven_entropy_cap_497.py`, `verify_seven_qubit_spinflip_spectral_geometry.py` |
| AH-1, AH-5 | `verify_seven_entropy_competitor_band.py` |
| AH-6 | `rational_intervals.py`, `verify_seven_entropy_joint_majorant.py` |
| E7-K0–E7-K3 | `certify_seven_entropy_cubic.py`, `certify_seven_entropy_family.py`, `certify_seven_entropy_ranks.py`, `data/seven_entropy/`, `verify_seven_entropy_modules.py` |
| E7-1 | `certify_seven_entropy_repair.py`, `data/seven_entropy/`, `verify_seven_entropy_compact.py` |
| TCB-0–TCB-4 | `certify_seven_qubit_three_copy_bound.py` |
| TCN-1–TCN-3 | `explore_seven_qubit_three_copy_sdp.py` |
| FC-0–FC-4 | `certify_seven_qubit_four_copy_bound.py`, `four_copy_certificate_n7.json` |
| NS-0–NS-4 | `certify_seven_qubit_four_copy_not_sharp.py`, `four_copy_primal_n7.json` |
| FS-0–FS-2 | `explore_seven_qubit_four_copy_sdp.py` |
| S5-1, Y5-1 | `verify_five_copy_site_algebra.py`, `verify_five_copy_young_blocks.py` |
| IK5-1–IK5-4, PK5-1–PK5-4 | `certify_ico_fivecopy_allv_faces.py`, `certify_ico_fivecopy_g34_42.py`, `certify_ico_fivecopy_g34_51.py`, `certify_ico_fivecopy_g34_6.py`, `certify_ico_fivecopy_g34_7.py`, `certify_ico_fivecopy_physical_faces.py`, `certify_ico_fivecopy_unflipped_oneflip_faces.py`, `verify_five_copy_candidate_basis.py` |
| DP5-1 | `verify_five_copy_probe_dimension.py` |
| FP5-1 | `certify_five_copy_repair.py` |
| FP5-2 | `verify_five_copy_scalar_probe.py` |
| FP5-3 | `verify_five_copy_evaluator_crosscheck.py` |
| FP5-4 | `verify_five_copy_repair_coordinates.py` |
| TT-1 | `verify_five_copy_column_parity.py` |
| FO-1 | `verify_icosahedral_family_orbit_data.py` |
| LH-1–LH-4 | `certify_icosahedral_full_hessian_exact.py` |
| QB-1–QB-3 | `verify_icosahedral_quantitative_basin.py` |
| EO-1 | `verify_icosahedral_label_orbits.py` |
| MO-1, MO-2 | `verify_moment_orbit_frame.py`, `verify_moment_orbit_pair_axes.py` |
| SF-1, SF-5 | `verify_sextic_focus_hierarchy.py` |
| HI-1–HI-4 | `verify_sextic_molien_series.py` |
| LP-0–LP-2, GS-1–GS-3 | `verify_balanced_purity_lp_graph_states.py` |
| LP-14, LP-15, GS-14, GS-15, LP-13, QP-13 | `verify_balanced_purity_13_15.py` |
| LP-19, GS-19, NG-1–NG-3 | `verify_balanced_purity_19.py`, `verify_nineteen_graph_improvement.py` |
| SW-0–SW-4 | `certify_three_copy_schur_weyl_bounds.py`, `three_copy_schur_weyl.py`, `three_copy_sw_certificates.json` |
| TA-1 | `explore_three_copy_all_n.py` |
| E13L-1–E13L-2 | `certify_thirteen_entropy_local.py`, `thirteen_entropy_local.json` |
| GL-1, GL-2 | `verify_graph_purity_16_17.py` |
| E18-1–E18-4 | `verify_balanced_purity_18.py` |
| TY-1–TY-2 | `verify_spinflip_typeII_16_20.py` |
| ST-1–ST-6 | `verify_spinflip_tradeoff.py` |
| ST-7, ST-8 | `verify_spinflip_tradeoff_odd.py` |
| ST-9 | `verify_odd_spinflip_witness.py` |
| SDS-1–SDS-3 | `verify_spinflip_doubled_selection.py` |
| CC-1 | `verify_complex_conference_sos.py` |
| CP-4 | `verify_conference_phase_rigidity.py` |

Helper modules and generators (imported by the scripts above or named in the paper; no tag of their own): `entropy_amplitude_hessian.py`, `entropy_code_tools.py`, `five_copy_allv_young_models.py`, `five_copy_candidate_data.py`, `five_copy_compact_evaluation.py`, `five_copy_exact_kernel.py`, `five_copy_physical_faces.py`, `five_copy_probe_evaluation.py`, `five_copy_probe_models.py`, `five_copy_repair_coordinates.py`, `five_copy_symmetric_models.py`, `five_copy_unflipped_oneflip_faces.py`, `five_copy_young42_model.py`, `five_copy_young51_model.py`, `four_copy_exact.py`, `four_copy_relaxation.py`, `gen_critical_side_jets.py`, `make_fig_cuts.py`, `make_fig_icosahedron.py`, `make_figures.py`, `renyi_dual_data.py`, `search_four_copy_certificate.py`, `search_three_copy_sw_certificate.py`, `seven_entropy_coefficients.py`, `seven_entropy_compact_evaluation.py`, `seven_entropy_cubic.py`, `seven_entropy_data.py`, `seven_entropy_evaluation.py`, `seven_entropy_joint.py`, `seven_entropy_joint_maps.py`, `seven_entropy_models.py`, `seven_entropy_source_evaluation.py`, `thirteen_fourcopy_gauss.py`, `three_copy_relaxation.py`.
The `explore_*` scripts give numerical evidence only (tags NV, TCN, FS, TA) and back no proof. `verify_supplement_restated.py` checks that the supplement restates
the paper's statements verbatim; it has no tag.
