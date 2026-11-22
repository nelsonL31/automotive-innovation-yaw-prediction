# Machine-readable manuscript tables

Tables 1–4 reproduce the main manuscript tables. S1–S27 are additional tables in this repository; the S numbers identify the archived analyses. The UTF-8 CSVs retain the original row order, column order, displayed precision and units. Cells spanning multiple columns or rows repeat their value so spreadsheet readers can reconstruct the headers. In equation cells, `^(...)` marks a superscript and `_(...)` a subscript.

The [index](index.json) gives the caption and size of every table. The main manuscript describes the study design. The [KIT protocols](../analysis/README.md), [planned-steering protocol](../evidence/README.md) and [sensitivity and training-duration data](../analysis/diagnostics/README.md) give the conditions for the corresponding numerical comparisons. Some tables include published comparator values, dataset descriptions or protocol decisions alongside this study’s own results. Interpret those cells using their original labels and source citations. The values have the precision reported in the paper. The planned-steering files in the evidence directory retain additional precision for recalculation.

ODE-PINN, PDE-PINN and the joint ODE–PDE PINN denote the three trajectory models. PGNN denotes a parameter-estimating network. The matched and reference PGNNs on KIT and LMSD are distinguished in the tables. For the racing comparison, published DDM values come from the reference paper and reproduced PGNN values come from this study’s implementation of that model.

## Source keys for the auxiliary records in S22

- S1. [ReV-StED dataset](https://doi.org/10.5281/zenodo.15270060).
- S2 and S3. [nuScenes dataset paper](https://doi.org/10.1109/CVPR42600.2020.01164) and [CAN bus expansion documentation](https://www.nuscenes.org/tutorials/can_bus_tutorial.html).
- S4. [TartanDrive 2.0 dataset paper](https://doi.org/10.1109/ICRA57147.2024.10611265).
- S5. [Experimental Sensor Data from Vehicles for Dynamic Vehicle Modeling, version 3](https://doi.org/10.17632/x7n6jnjh36.3).
