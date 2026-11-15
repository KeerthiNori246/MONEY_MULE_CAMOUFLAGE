# The Camouflage Paradox in Money Mule Fraud Graphs: Independent Structure--Feature Drift Signals

This repository contains the experimental implementation of **"The
Camouflage Paradox in Money Mule Fraud Graphs: Independent
Structure--Feature Drift Signals."**

The study investigates how adversarial money mule camouflage differs
from benign evolution in financial transaction graphs. It proposes two
complementary signals --- **Structural Coordination Intensity (SCI)**
and **Embedding Geometry Signals (EGS)** --- to characterize structural
and embedding-space changes over time.

## Publication

**Accepted and Presented:** International Conference on High Performance
Computing and Artificial Intelligence (**HPCAI 2026**).

The paper is currently undergoing the post-conference publication
process and is expected to appear in Springer's *Studies in Autonomic,
Data-driven and Industrial Computing* book series, which is
Scopus-indexed.

## Overview

Financial transaction graphs evolve naturally as legitimate users change
their transaction behavior. However, money mules may deliberately
manipulate transaction patterns and graph connections to conceal
fraudulent activity.

Although both processes introduce drift, they can produce fundamentally
different signatures.

This work introduces two complementary measurements:

-   **Structural Coordination Intensity (SCI):** Captures coordinated
    structural changes through temporal motif evolution and hub
    redistribution dynamics.
-   **Embedding Geometry Signals (EGS):** Captures changes in embedding
    space through prototype center movement, cluster dispersion, and
    anisotropy.

Together, these signals help distinguish benign drift from adversarial
drift while examining whether structural and feature-based changes
affect them independently.

## Key Contributions

1.  **Structural Coordination Intensity (SCI):** Combines temporal motif
    evolution and hub redistribution into a unified structural drift
    signal.
2.  **Embedding Geometry Signals (EGS):** Combines prototype movement,
    cluster dispersion, and anisotropy to characterize embedding-space
    evolution.
3.  **Controlled Causal Ablation:** Evaluates structure-only,
    feature-only, and combined perturbations at multiple intensity
    levels (5%, 15%, and 30%).
4.  **Signal Independence:** Demonstrates that SCI is invariant to
    feature-only perturbations, while EGS responds to both structural
    and feature-based changes through separate pathways.

## Experimental Setup

**Dataset:** LOL Bank Pvt. Ltd. Bank Transaction Fraud Detection dataset
(Kaggle).

The transaction data is represented as heterogeneous transaction graphs
with four directed edge types:

-   Customer → Transaction
-   Transaction → Merchant
-   Transaction → Device
-   Transaction → Location

The experiments examine benign and adversarial drift, alongside three
controlled perturbation conditions:

| Perturbation | Description |
|---|---|
| **Structure-only** | Rewires transaction-to-merchant edges while keeping node features unchanged. |
| **Feature-only** | Perturbs transaction amounts using multiplicative Gaussian noise while keeping graph topology unchanged. |
| **Combined** | Applies structural rewiring and feature perturbations simultaneously. |

Each perturbation is evaluated at low (5%), medium (15%), and high (30%)
intensity.

## Repository Workflow

The implementation is organized around three experimental components:

-   **SCI:** Temporal motif evolution, hub redistribution, unified SCI
    computation, and spike detection.
-   **EGS:** Embedding geometry time-series computation and unified EGS
    analysis.
-   **Coupling and Independence Analysis:** Generation of controlled
    drift baselines, embedding generation, and SCI/EGS comparisons
    across perturbation conditions.

## Requirements

-   Windows with PowerShell
-   Python and the dependencies listed in `requirements.txt`
-   PyTorch (required only when regenerating embeddings)

## Setup

Run the following commands in **PowerShell**, from the repository root.

The baseline graphs, embeddings, and model files are already present in
this checkout.

``` powershell
cd C:\MONEY_MULE_CAMOUFLAGE\
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install numpy matplotlib
```

## Running the Experiments

Execute the following scripts in the specified order.

### 1. Structural Coordination Intensity (SCI)

Run motif and hub calculations before combining them, followed by spike
detection.

``` powershell
python src\temporal\sci\temporal_motif_evolution.py
python src\temporal\sci\hub_redistribution.py
python src\temporal\sci\compute_sci.py
python src\temporal\sci\sci_spike_detection.py
```

### 2. Embedding Geometry Signals (EGS)

Calculate the component time series, combine them into unified EGS, and
generate spike analysis.

``` powershell
python src\graphs\egs\egs_timeseries.py
python src\graphs\egs\compute_unified_egs.py
python src\graphs\egs\egs_spike_detection.py
```

### 3. Coupling and Independence Analysis

#### Step A: Generate controlled drift baselines

Create the structure-only, feature-only, and combined-drift baselines.

``` powershell
python src\temporal\egs\create_structure_drift.py
python src\temporal\egs\create_feature_drift.py
python src\temporal\egs\create_combined_drift.py
```

#### Step B: Generate embeddings

If you regenerate the baselines, regenerate their corresponding
embeddings. This step requires PyTorch and the existing model files.

``` powershell
python -m pip install torch
python src\temporal\egs\embed_snapshots_drift.py
python src\temporal\egs\embed_snapshots_feature_drift.py
python src\temporal\egs\embed_snapshots_combined_drift.py
```

#### Step C: Compute SCI and EGS comparisons

Run the SCI comparisons for all three drift types:

``` powershell
python src\temporal\egs\compute_sci_drift.py
python src\temporal\egs\compute_sci_feature_drift.py
python src\temporal\egs\compute_sci_combined_drift.py
```

Then run the corresponding EGS comparisons:

``` powershell
python src\temporal\egs\compute_egs_drift.py
python src\temporal\egs\compute_egs_feature_drift.py
python src\temporal\egs\compute_egs_combined_drift.py
```

**Important:** If you use the existing baseline graphs and embeddings
included in this checkout without regenerating them, skip Steps A and B.
Proceed directly to Step C.

## Key Findings

The experiments indicate that:

-   **Structural separability:** Adversarial drift produces more
    persistent SCI oscillations, whereas benign drift exhibits
    comparatively smooth structural evolution.
-   **Embedding-space instability:** Adversarial drift produces
    persistent prototype movement and anisotropic distortion, unlike the
    comparatively stable geometry observed under benign drift.
-   **Feature independence of SCI:** Feature-only perturbations do not
    trigger SCI, even at the highest tested intensity.
-   **Complementary signals:** EGS responds to both structural and
    feature perturbations, while SCI captures coordinated structural
    changes. Their different sensitivities provide complementary
    coverage under the tested conditions.

## Limitations

The experiments use a single transaction graph dataset and simulated
adversarial perturbations rather than observed real-world attacks.
Further evaluation across datasets, fraud types, and adaptive
adversarial settings is necessary.

## Authors

**Keerthi Nori**\
**Dr. M. Nagaraju**\
Institute of Aeronautical Engineering, Hyderabad, India

## Reference

Nori, K., and Nagaraju, M. *The Camouflage Paradox in Money Mule Fraud
Graphs: Independent Structure--Feature Drift Signals.* Accepted and
presented at HPCAI 2026. Expected publication in Springer's *Studies in
Autonomic, Data-driven and Industrial Computing* book series;
post-conference publication process ongoing.
