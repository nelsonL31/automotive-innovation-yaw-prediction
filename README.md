# How steering information affects the relative accuracy of physics-informed and data-only models for short-horizon yaw-rate prediction

This repository contains the [KIT model definitions and vehicle configuration](models/README.md), inputs and code for the seven figures, and the derived results for the measured-vehicle and simulation comparisons. The data below can be used to check how steering availability changes model rankings, how stiffness estimates transmit disturbances, and how open-loop prediction compares with closed-loop lap completion.

## Data in the article

The [table guide](tables/README.md) and [index](tables/index.json) identify the four manuscript tables and 27 supporting CSV tables archived here. The main tables retain the precision shown in the manuscript. The [scale-racing lap records](analysis/closed_loop/README.md) give the outcome, boundary excursions, solver deadline misses and stopping reason for each of the 12 simulated attempts in Table 4. The [fixed-stiffness matrices](analysis/stability/README.md) cover eight axle stiffness settings at six speeds. The [diagnostic data](analysis/diagnostics/README.md) include the KIT origin-level sensitivities, full-scale racecar seed-level errors and the 400- versus 2,500-epoch comparisons.

## Verify the numerical results

From the repository root, run:

```sh
python verify_evidence.py
python verify_diagnostics.py
```

Both commands use the Python standard library. The diagnostics check recomputes 48 straight-running Jacobians and the reported critical speed, reconstructs sensitivity statistics and racecar comparisons from the derived records, and checks the four lap-completion counts in Table 4. The planned-steering check verifies ten archived file hashes, checks all combinations of scenario, model, fit seed and input condition, and reconstructs the yaw-rate comparisons in `evidence/paired_trajectories.csv` and `evidence/summary.json`. Steering-path distances are carried forward from the paired table. Recomputing those distances from individual wheel angles requires the archived simulation arrays.

In the 12 simulated trajectories, the mean endpoint yaw-rate RMSE for the data-only and ODE-constrained models, respectively, was 0.13074 and 0.06691 rad/s with the current wheel angle held. With the actuator-mapped plan available at the forecast origin, the corresponding errors were 0.10258 and 0.01818 rad/s. Each model is averaged over the same three fit seeds before trajectories are given equal weight.

## Redraw the figures

See [figure scripts and inputs](figures/README.md) for the plotting environment, files and commands. The figure directory covers the manuscript's figures, including results from measured vehicles and simulations. The evidence directory provides the extra numerical detail for the planned-steering comparison.

## Diagnose input and model sensitivity

The [analysis directory](analysis/README.md) contains the source and protocol records for the held-versus-replayed steering comparison, steering-excursion groups, the tyre-plant simulation, and the endpoint-gain and persistent-bias checks. Run `python analysis/check_figure_inputs.py` from the repository root to verify the archived inputs for Figures 2, 4 and 5 against the release fingerprints.

## Run the archived inference workflow

The two scripts in [evidence/reproduce](evidence/reproduce/) are project research code. The archived execution used Python 3.12.3, NumPy 2.1.3 and PyTorch 2.8.0+cu128. The primary evaluation ran on a single CPU thread. The subsequent device check used a GeForce RTX 5090 with TF32 disabled.

The [model directory](models/README.md) contains the project’s KIT network definitions, vehicle contract, integration settings and frozen cohort manifest. It documents which files are byte-identical to the registered versions and how workstation paths were made portable in two configuration files. The upstream network builder is linked at its pinned revision. Running predictions again requires the six archived checkpoints and the simulation inputs described in the [evidence guide](evidence/README.md). The checkpoints and full per-origin simulation and prediction arrays are held in the research archive. For third-party vehicle benchmarks, the release contains the study's derived results. Obtain original vehicle logs directly from their providers under their respective terms. Numerical libraries and upstream model or dataset software retain their own distribution terms.

## Original benchmark sources

* KIT Multi-Surface. T. Schulz and F. Snobar, *Multi-Surface Driving Maneuvers: A Dataset from Standstill to Vehicle Dynamics Limits*. Karlsruhe Institute of Technology, 2026. [Dataset record](https://publikationen.bibliothek.kit.edu/1000192628)
* LMSD concept car. M. Viehweger et al., *Automotive Experimental Reference Data Set: LMSD Concept Car, Lommel (BE), 2023*. KU Leuven RDR, 2024. [Dataset record](https://doi.org/10.48804/PHMF9D)
* Racing benchmarks. The Deep Dynamics release supplies the AV-21 full-scale Putnam Park and LVMS records and the ETHZ and ETHZMobil BayesRace scale-vehicle simulation files. [Upstream data inventory](https://github.com/linklab-uva/deep-dynamics#processing-data)

