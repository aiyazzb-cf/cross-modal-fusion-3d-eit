<div align="center">

# Cross-Modal Fusion 3D EIT Framework

**Cross-Modal Fusion 3D Electrical Impedance Tomography Framework for Spectral-Complementarity Reconstruction Based on Optimal Information Supply and Physical Priors**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)]()
[![MATLAB R2019b+](https://img.shields.io/badge/MATLAB-R2019b%2B-red.svg)]()
[![EIDORS](https://img.shields.io/badge/EIDORS-3.9%2B-green.svg)](http://eidors3d.sourceforge.net)

Official implementation of the paper *"A Cross-Modal Fusion 3D EIT Imaging Framework Based on Optimal Information Supply and Physical Priors"* (under review).

</div>

---

## Overview

Three-dimensional Electrical Impedance Tomography (3D EIT) suffers from a fundamental information bottleneck: raw boundary voltages retain the complete measurement spectrum but lack spatial interpretability, while regularized reconstructions provide structural priors yet discard high-frequency details due to inherent low-pass filtering.

This framework exploits the **spectral complementarity** between the two modalities through a dual-domain fusion architecture composed of three coupled modules:

| Module | Full name | Role |
|---|---|---|
| **Dual-M SMP** | **D**ual-**M**aximum **S**timulation–**M**easurement **P**attern | Front-end channel selection |
| **UREIT** | **U**niform **R**econstruction for **EIT** | Image-domain anchor generation |
| **VIFI-Net** | **V**oltage–**I**mage **F**usion **I**maging **N**etwork | Cross-modal nonlinear fusion |

### 1. Dual-M SMP — optimal information supply
Selects the stimulation–measurement channels (SMCs) with **maximal total Measurement Sensitivity (MS) under a strict independence constraint**:

- **Random feature mapping**: each SMC is mapped to a low-dimensional feature vector by a random matrix `E`, encoding the superposition theorem (row differences) and the reciprocity theorem (Hadamard-product commutativity) with probability-1 consistency.
- **Matroid theory**: the greedy selection (Rado–Edmonds theorem) provably attains the maximum-weight basis — i.e., the Max-I SMP with maximal total MS.

### 2. UREIT — spatial uniformity of the inverse problem
Analytically couples the **compensation exponent `c`** with the regularization parameter `λ`:

```math
c_{\mathrm{opt}}(\lambda) = \frac{\ln\frac{d_{\min}+\lambda}{d_{\max}+\lambda}}{\ln\frac{d_{\min}}{d_{\max}}} - 1,
```

building an adaptive diagonal compensator `D^c` that homogenizes the inverse-problem sensitivity and produces an image-domain anchor with robust low-frequency structure and spatially balanced response.

### 3. VIFI-Net — cross-modal fusion
Integrates:

- a **reciprocity-theorem-based voltage-to-matrix mapping (V2M)**,
- **asymmetric cross-domain attention** (image features query voltage tokens),
- **Sensitivity-aware spatial attention** provides local nonlinear compensation for the inverse problem sensitivity,

for nonlinear multi-scale fusion of the two heterogeneous modalities.

---

## Key Results

- **Dual-M SMP** improves average MS by **> 85%** across six numerical models.
- **UREIT** improves reconstruction uniformity relative to Tikhonov regularization at **93%** of the spatial locations.
- **VIFI-Net**: best MAE / PSNR / SSIM / GMSD among compared methods.

---

## Repository Layout

```
opencode/
├── matlab/                          # Dual-M SMP + UREIT (EIDORS-based)
│   ├── dual_m_smp.m                 #   Dual-M SMP construction (main entry)
│   ├── build_feature_matrix.m       #   random feature matrix Z = (S·E)⊙(M·E)
│   ├── rank_smc_set.m               #   independence rank of an SMC set
│   ├── random_feature_matrix.m      #   random feature-mapping matrix E
│   ├── channel_sensitivity.m        #   per-channel measurement sensitivity
│   ├── find_independent_smcs.m      #   rref-based independent-channel selection
│   ├── smc_list_to_stimulation.m    #   SMC list → EIDORS stimulation structs
│   ├── ureit_reconstruction_matrix.m #  UREIT reconstruction matrix (NF-matched λ)
│   ├── elem_measure.m               #   element area (2D) / volume (3D)
│   ├── voltage_to_matrix.m          #   V2M: responses → electrode-pair matrix
│   ├── interpolate_by_electrode_layout.m #  layout-based completion of Y
│   ├── main.m                       #   end-to-end demo (SMP → UREIT → V2M)
│   └── FEMs/                        #   EIDORS forward models (cylinder/head/thorax)
└── python/
    └── VIFI-Net/
        └── model.py                 # VIFI-Net implementation (PyTorch)
```

---

## Requirements

| Language | Dependencies |
|---|---|
| **MATLAB** | MATLAB **R2019b+** · [EIDORS](http://eidors3d.sourceforge.net) (only core functions: `mk_image`, `mk_stim_patterns`, `calc_jacobian`) — **no extra toolboxes** |
| **Python** | Python **3.9+** · PyTorch **1.13+ / 2.x** |

The `FEMs/*.mat` files are EIDORS forward models (8/16/32/64-electrode cylinders, head, thorax) used in the paper.

---

## Quick Start

### MATLAB — Dual-M SMP + UREIT

```matlab
% Load a 32-electrode cylinder forward model (provides struct fmdl)
load(fullfile('matlab', 'FEMs', 'cylinder32.mat'));
sigma = ones(size(fmdl.elems, 1), 1);       % homogeneous background

% 1) Dual-M SMP: Max-I SMP of maximal total MS
smp = dual_m_smp(fmdl, sigma);              % (M_max x 4): [s- s+ m- m+]

% 2) UREIT: adaptive-uniform reconstruction matrix
img = mk_image(fmdl, sigma);
img.fwd_model.stimulation = mk_stim_patterns(numel(fmdl.electrode), 1, ...
    [0 1], '{ad}', {'meas_current'}, 1e-2);
jacobian = calc_jacobian(img);
[reconMatrix, lambda] = ureit_reconstruction_matrix(fmdl, jacobian, 1, ...
    'verbose', true, 'randomSeed', 42);     % reproducible noise samples
```

`main.m` runs the full pipeline end-to-end.

### MATLAB — V2M (voltage-to-matrix mapping)

Map the Dual-M SMP voltage responses onto the dense electrode-pair matrix
that VIFI-Net consumes:

```matlab
% Sensitivity matrix on the Dual-M SMP channel set
img.fwd_model.stimulation = smc_list_to_stimulation(smp, 1e-2, numel(fmdl.electrode));
jacobian = calc_jacobian(img);
y = jacobian * perturbation;               % simulated voltage change (n x 1)

% Electrode centroids
electrodes = zeros(numel(fmdl.electrode), 3);
for i = 1:numel(fmdl.electrode)
    electrodes(i, :) = mean(fmdl.nodes(fmdl.electrode(i).nodes, :), 1);
end

Y = voltage_to_matrix(smp, electrodes, y); % (P x P x T), single precision
```

Entries are filled by reciprocity symmetry and completed by
`interpolate_by_electrode_layout`.

### Python — VIFI-Net

```python
import torch
from model import VIFINet

net = VIFINet(sensitivity_map,          # (D, H, W) prior sensitivity volume
              token_dim=128,
              use_film=True)

Y      = torch.randn(B, 1, H, W)        # voltage-difference matrix (Dual-M SMP)
y      = torch.randn(B, volt_dim)       # raw voltage vector (e.g. 464 for 32 electrodes)
X_init = torch.randn(B, 1, D, H, W)     # UREIT initial image (even D, H, W)

out = net(Y, y, X_init)                 # (B, 1, D, H, W) refined reconstruction
```

**Shape convention**: `X_init` and `sensitivity_map` must have **even** spatial
dimensions — the stride-2 down-sampling and `Upsample(scale_factor=2)` skip
connections differ by one voxel for odd sizes.

**Architectural note**: the paper describes three down-sampling stages
(64 → 128 → 256 → 512); this implementation uses two (64 → 128 → 256), i.e.
one stage shallower, as an intentional simplification.

---

## Module Usage

### `dual_m_smp(fmdl, sigma, options...)`

Name-Value options:

| Option | Default | Description |
|---|---|---|
| `currentAmp` | `1e-2` | stimulation current amplitude (A) |
| `independenceMethod` | `"gaussian"` | `"gaussian"`: batched rank test + bisection (fast) · `"rank"`: per-channel greedy |
| `forwardMethod` | `"adjacent"` | `"adjacent"`: superposition from SMP_ads (fast) · `"direct"`: batched Jacobian solves |
| `measurementSet` | `"reciprocity"` | `"reciprocity"`: reduced universe (reciprocity) · `"permutation"`: all polarities |
| `randomSeed` | `[]` | seed for the random matrix `E` (reproducible; rng restored on exit) |

Output: `(M_max x 4)` matrix, rows `[s-  s+  m-  m+]`, with
`M_max = N(N-3)/2` independent channels.

### `ureit_reconstruction_matrix(fmdl, jacobian, noiseFigure, options...)`

| Option | Default | Description |
|---|---|---|
| `lambda` | `[]` | explicit regularizer; when given, the NF search is skipped |
| `noiseSamples` | `500` | Monte-Carlo noise vectors for the NF estimate |
| `tolerance` | `[]` | NF matching tolerance (auto: 1e-3, relaxed to 1e-2 for >3000 elements) |
| `verbose` | `false` | print the λ search trace |
| `randomSeed` | `[]` | seed for the noise vectors (reproducible) |

Outputs: `[RM, lambda]` — the compensated reconstruction matrix and the
NF-matched regularization parameter.

> **Large models**: each NF evaluation multiplies `RM` by 500 noise vectors;
> for very large meshes use `'noiseSamples', 100` for fast experiments.

### `voltage_to_matrix(smp, electrodeCoords, y)`

| Argument | Description |
|---|---|
| `smp` | (n x 4) Dual-M SMP channel list `[s- s+ m- m+]` |
| `electrodeCoords` | (N x 2/3) electrode coordinates |
| `y` | (n x T) voltage responses |

Output: `Y` — (P x P x T) dense `single` matrix over the P unique
electrode pairs, with reciprocity symmetry; missing entries are completed by
`interpolate_by_electrode_layout` (position / direction / length similarity
kernels with automatic `sigma_d`, `sigma_L`).

### `VIFINet`

```python
VIFINet(sensitivity_map, token_dim=128, use_film=True)
```

See the docstrings in `model.py` for details.

---

## Reproducing the Paper

1. Generate / load an EIDORS forward model (see `FEMs/`).
2. Build the Dual-M SMP → compute its Jacobian → `y`.
3. Build the UREIT reconstruction matrix → obtain the initial image `X_init`.
4. Map the Dual-M SMP responses onto the electrode-pair matrix with V2M → `Y`.
5. Train VIFI-Net on `(Y, y, X_init)` triplets with the supervision of the
   ground-truth conductivity image.

---

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{DUALM_SMP_UREIT_VIFINET,
  title   = {Cross-Modal Fusion 3D Electrical Impedance Tomography Framework for Spectral Complementarity Imaging Based on Optimal Information Supply and Physical Priors},
  author  = {Zhao, Zhibo and Feng, Fu and Yang, Lin},
  year    = {2026},
  note    = {under review}
}
```

---

## License

This project is licensed under the **MIT License** — see the
[LICENSE](LICENSE) file.

---

## Acknowledgments

- The forward-problem computations build on the open-source
  [EIDORS](http://eidors3d.sourceforge.net) toolbox for electrical impedance
  tomography.
