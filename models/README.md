# KIT lateral-prediction models

`kit_models.py` contains the project’s two-state single-track equations, fourth-order Runge–Kutta update, neural architectures and training objectives. The state comprises lateral velocity at the centre of gravity and yaw rate. Each input supplies the current longitudinal speed and steering angle together with 51 time-aligned history samples. The model advances the state by 0.02 s.

## Model definitions

| Factory key | Prediction and objective |
| --- | --- |
| `data` | A neural flow map predicts the lateral states using the data loss. |
| `ode` | The same flow map also matches its time derivative to the vehicle vector field at the predicted state. |
| `pde` | The same flow map also matches its time derivative to its initial-state Jacobian applied to the vector field at the initial state. |
| `joint` | The same flow map uses the mean of the two physics losses in addition to the data loss. |
| `pgnn_matched` | A two-hidden-layer parameter network estimates bounded front and rear cornering stiffnesses for the single-track integrator. |
| `pgnn_ref` | The project’s adapted parameter model uses the reference GRU network builder with a project-specific two-stiffness output head. |

For the flow-map models, the network predicts a scaled increment added to the initial state and multiplied by prediction time. This construction gives the measured initial state at zero time. `objective` implements the data and equation losses, and `factory` constructs each arm.

## Required upstream builder

The archived module imports `official_build_network` at module load time. Use PyTorch and install the upstream dependencies required by the builder. Obtain the pinned source from the Deep Dynamics authors and save it beside `kit_models.py` as `official_build_network.py`. [Upstream source at the pinned commit](https://github.com/linklab-uva/deep-dynamics/blob/fe50011c65a0679bb79a077ec28710bd373a1ea1/deep_dynamics/model/build_network.py) · [Raw file](https://raw.githubusercontent.com/linklab-uva/deep-dynamics/fe50011c65a0679bb79a077ec28710bd373a1ea1/deep_dynamics/model/build_network.py)

The upstream file uses LF line endings and has SHA-256 `57348da275952319960f1e22aad0700859f567cdadfe1aa6670b74de5764e9ed`. The archived copy used CRLF line endings and has SHA-256 `c781300cdc1775b985d22cfa34677959a8a4937bebc59fc291fcde968be7d032`. Converting the 49 line endings reproduces the archived bytes. Either encoding gives the same Python source.

## Frozen settings and model inputs

- `kit_contract.json` records the input and output order, geometry, physical parameters, regularisation and network settings. Its relative `source_root` identifies where locally prepared training inputs would be placed. The file records an early engineering stage. The completed training runs are listed in the later `frozen_cohort.json`.
- `integration_result.json` contains the training-derived normalisation, the integration-resolution comparison and the selected eight substeps per 0.02 s model step. The three input records are identified by filename and content hash.
- `frozen_cohort.json` records final checkpoint SHA-256 values, run status, epochs and update counts. A hash identifies a trained checkpoint. Re-evaluating a trained predictor requires the corresponding weights and appropriately prepared records.
- `model_provenance.json` pairs the hashes of the distributed files with those recorded in [the frozen protocol](../evidence/protocol.json). The Python module and cohort manifest match the registered bytes. The two settings files retain their model values and replace the original machine-specific directories with relative paths or basenames.

The six factory keys were checked with CPU PyTorch 2.8.0 using synthetic inputs. Each returned a finite three-by-two prediction, a finite total loss and finite parameter gradients on a single backward pass. This checks the model interface; the reported experimental values come from the frozen runs.

## Example forward pass

From this directory, after supplying the upstream builder:

```python
import json
from pathlib import Path

import torch

from kit_models import factory

settings = json.loads(Path("integration_result.json").read_text())
model = factory("ode", settings["normalization"]).eval()
state = torch.tensor([[0.25, 0.20]])           # lateral velocity, yaw rate
control = torch.tensor([[10.0, 0.02]])        # longitudinal speed, steering
history = torch.zeros(1, 51, 4)               # illustrative synthetic history
with torch.no_grad():
    next_state = model(state, control, history)
```

This example instantiates the model with its initial weights.
