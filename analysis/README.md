# KIT analysis provenance for Figures 2, 4 and 5

These files document the analyses that produced the plotted numerical inputs for the Automotive Innovation manuscript. The accompanying [figure package](../figures/README.md) redraws the figures directly from the distributed CSVs.

## Verify the plotting inputs offline

Run `python analysis/check_figure_inputs.py` from the repository root. This standard-library check compares every field of four plotting tables with fingerprints calculated independently from the completed analyses on the research archive. The comparison sorts complete rows and represents numbers with 14 significant digits. It covers all 10 input-condition rows, 6 steering-excursion rows, 72 tyre/input configurations and 1,458 local gains. The figure package separately checks the exact CSV byte digests. With access to the original study archive, `--publication <publication-analysis-directory> --execution <KIT-evaluation-directory>` also reconstructs these fingerprints from the archived summary JSONs without running inference.

| Manuscript result | Analysis source and unit |
| --- | --- |
| Fig. 2a–c | `fig2_factor/kit_factor.py` independently holds or replays speed and steering on the same sampled drives. It averages seed-specific drive errors before aggregating across drives. The archived GPU parity run completed all comparisons. The earlier CPU attempt stopped on the third drive when its results exceeded the registered numerical tolerance. |
| Fig. 2d | `fig2_excursion/analyse.py` groups previously saved 0.6 s prediction errors by maximum absolute deviation of future steering from its origin value at offsets 0 through 29. The figure uses the held-input, cross-surface, yaw-rate result for each of the three steering bins. The final bin contains three drives. This retrospective grouping uses observed future steering only to label the groups. |
| Fig. 4 | `fig4_tyre_plant/independent.py` is the completed baseline-interface repair. It generates 24 independent sets of input phases and pairs each set across two tyre laws, nine front/rear stiffness combinations and four input conditions. The completed archive records 13,824 metric rows from 432 phase–plant–stiffness trajectories. The original incomplete attempt supplied no figure results. |
| Fig. 5 | `fig5_gain/audit.py` propagates derivatives through the 30 steps, history feedback and learned axle stiffness. `summarize.py` aggregates observations and `bias_projection.py` projects the Jacobian onto persistent state-offset directions. The projection was specified after the main analysis. The 1,458 plotted values span two model types and three perturbation conditions on the same origins. |

## Original and distributed files

[provenance.json](provenance.json) records the exact SHA-256 of each archived input script or protocol, its relative source location and the SHA-256 of its distributed copy. Checksums for upstream result and completion files identify the precise evidence version. The archived protocol for Fig. 4 identifies the repaired script. The original Fig. 2 GPU protocol contains a guard description naming the earlier CPU comparison. Its amendment, runtime and executed code establish the GPU backend for the successful run.

The distribution replaces fixed server input directories with `KIT_FROZEN_ROOT` and uses the analysis script directory for outputs. In the Fig. 2 protocol, symbolic labels replace absolute directory prefixes on five code-digest keys. Protocol content otherwise follows the archive. The completed program registered for Fig. 4 checks its own source hash. Its archival protocol identifies the original file bytes. A relocated rerun must generate a fresh local protocol for the distributed code using its original `--freeze` entry point. Keep `registered_protocol.json` as the historical reference. Its byte digest and run timestamp do not describe the relocated copy.

## Dependencies and data access

The frozen runs used Linux x86-64, Python 3.12.3, NumPy 2.2.2 and PyTorch 2.8.0+cu128. Figure 2a–c used a GeForce RTX 5090, the original deterministic PyTorch switches and the original archived comparison tolerances. Figure 4 ran on CPU. Figure 5 ran on CPU with double precision, single-threaded PyTorch and SciPy for Spearman rank correlation. The figure-reproduction package documents the separate, lightweight plotting environment.

For an authorised upstream audit, set `KIT_FROZEN_ROOT` to a **trusted, complete** KIT study evaluation workspace. Its required contents depend on the analysis:

- Figure 2 uses the verified original evaluator and channel adapter (`evaluate_kit.py`, `kit_data.py`, `kit_models.py`, and `export_validation/`), training-only normalisation, cohort records, frozen checkpoints, the fitted ARX coefficients, licensed KIT source records and their original saved predictions. The excursion calculation uses the previously stored predictions and official source records.
- Figure 4 requires exactly the original cohort, contract, integration metadata, network builder, model definitions, six final checkpoints and ARX coefficients identified in its archival protocol. The sampling and plant constants are in the distributed script.
- Figure 5 requires the original evaluation workspace, verified checkpoint cohort, valid source records and archived hold forecasts. The projection reads the Jacobians produced by the main audit. Detailed Jacobians and measured trajectory targets stay in the controlled archive.

Use a separate working copy for any new run. The scripts write results beside their source files. Figure 2a–c generates its own `protocol.json`; Figure 4 uses `--freeze` before running the relocated copy. The original evaluator may require the registered upstream build, dataset adapter and authorised records beyond the project files provided here. The release supports offline verification and figure reproduction, and documents the additional prerequisites for full numerical reruns.
