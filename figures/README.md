# Reproducing the seven manuscript figures

These scripts regenerate the English figures for *Steering information and tyre cornering stiffness in short-horizon vehicle yaw-rate prediction*. Each figure reads the CSV files in its own folder. The package includes project-generated simulation outputs, prediction-error summaries and computed sensitivities. The scripts run offline and require no vehicle telemetry, training checkpoints or GPU.

## Run

From the repository root, use a Python environment with the versions in `figures/requirements.txt`.

```shell
python -m pip install -r figures/requirements.txt
python figures/reproduce.py --output figure-output --verify
```

To draw selected figures, add their numbers, for example `--figures 2 5`. Each figure creates a PDF, an SVG and a PNG in the chosen output directory. An existing file with the same name in that directory will be replaced. The CSV inputs remain in their source directories.

## Data and interpretation

| Figure | Input files inside the figure folder | Contents and comparison unit |
| --- | --- | --- |
| 1 | `steering_trace.csv` | 45 sample times from one project-generated steering example, at 0.02 s intervals. Time zero is the forecast origin. Columns give the steering history, hold, original plan after actuator mapping and executed steering in radians. Empty fields identify the part of each series outside its time interval. The plot uses milliradians. |
| 2 | `input_conditions.csv`, `steering_excursion.csv` | Ten rows in the first file give three method means for each KIT surface group and four drive-specific rows of **ODE minus data-only** yaw-rate RMSE. The second file gives six method–excursion combinations, with drive counts and forecast-origin counts per seed. `sealed_id` identifies the same-surface drives and `sealed_surface` the cross-surface drives. |
| 3 | `paired_trajectories.csv`, `per_fit_trajectory.csv` | Twelve simulated trajectories comprise two friction settings, two steering manoeuvres and three shared phase blocks. The 216 rows cover 12 trajectories × two models × three fits × three steering inputs. A holds the wheel angle, B uses the issued plan with actuator mapping and C replays the executed angle. `gap_B_minus_A` is the change in the ODE-minus-data RMSE gap from A to B, in rad/s. Panel (c) displays this change in mrad/s. |
| 4 | `tyre_simulation.csv` | 72 configurations comprise two tyre laws, three front and three rear stiffness multipliers, and four input modes. Each row summarises the same 24 paired phases and reports yaw-rate RMSE in rad/s. Percentage change is 100 × (ODE error / data-only error − 1). |
| 5 | `local_gains.csv` | 1,458 dimensionless gains from two fitted model types and three perturbation conditions. Each condition and model contributes 135 origin–seed observations on five same-surface drives and 108 on four cross-surface drives. `gain` is the full derivative, `gain0` cuts the stiffness derivative path, and `gain_per_unit_stacked_norm` applies a persistent bias. Origins within a drive share the recorded trajectory. |
| 6 | `record_errors.csv`, `aggregate_errors.csv` | The first 18 rows cover six LMSD drives × three models. Each row gives the seed-averaged RMSE and maximum absolute error of three states at 0.6 s. Velocities are in m/s and yaw rate in rad/s. The second file contains 18 aggregate comparator–metric rows used to check the equal-drive means for the plotted methods. |
| 7 | `racing_errors.csv` | 144 rows cover three Putnam training records, two LVMS test records, three trajectory models and eight metrics from the 1,000-epoch cohort. Each row contains the method error and its paired reproduced PGNN error. The plot uses the percentage changes in one-step yaw-rate RMSE and 15-step final displacement error. LVMS-A and LVMS-B are historical reevaluations. |

The simulations in Figs. 1, 3 and 4 were generated for this study. Record names in the other folders identify the public datasets used for the project’s own model evaluations. The original telemetry and its redistribution terms remain with the source repositories. KIT is cited as Ref. 26, LMSD as Ref. 27, and the full-scale racing data and Deep Dynamics release as Refs. 28–29 in the article. [KIT Multi-Surface](https://publikationen.bibliothek.kit.edu/1000192628) · [LMSD Concept Car](https://doi.org/10.48804/PHMF9D) · [RACECAR](https://arxiv.org/abs/2306.03252) · [Deep Dynamics data release](https://github.com/linklab-uva/deep-dynamics/tree/main/deep_dynamics/data)

## Verification

Before drawing each figure, `reproduce.py` checks the SHA-256 digests of its CSV inputs against `reference.json`. Plotting scripts retain their numerical checks, including reconstruction of the Fig. 3 paired differences, the Fig. 6 drive means and the Fig. 7 percentage changes.

The optional `--verify` check renders each rebuilt PDF at 144 dpi with PDFium and compares its RGB pixels against a digest obtained from the corresponding final submission PDF. All seven figures matched in the recorded environment, which used Python 3.13.5, Matplotlib 3.10.5, NumPy 2.2.6 and pandas 2.3.2 on Windows with Arial installed. PDFium 153.0.7999.0 (pypdfium2 5.13.0) produced the archived reference digests. The JSON also records the hashes of the original submission PDFs. PDF metadata and dates may differ between exports. A different font installation or rendering library can change the pixel check even when the data and plots agree. Arial is not bundled.

The scripts correspond to the final artwork, including the versions of Figs. 2, 6 and 7 retained from the earlier analysis. The English export entry points, local filenames and output paths have been aligned to the manuscript figure numbers. This package reproduces the plots from derived data. Training and raw-record preprocessing are separate stages.

