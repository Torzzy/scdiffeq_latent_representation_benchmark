# A Latent Space Benchmark for Future Cell Prediction in Single-Cell RNA-seq

**Tom Dauvé — 2026**

Benchmarking PCA, scVI, VELOVI and FlatVI latent representations for future cell-state prediction from single-cell RNA-seq using a neural stochastic differential equation framework.

---

## Overview

Single-cell RNA sequencing (scRNA-seq) provides a high-dimensional and sparse representation of cellular states. Latent representations are commonly used to reduce this dimensionality while preserving biologically relevant information.

However, the choice of latent representation may affect the ability to model cellular dynamics and predict future cell states.

This project investigates how different latent representations influence future cell-state prediction using a neural stochastic differential equation (SDE) framework.

Four latent representations are benchmarked:

* **PCA**
* **scVI**
* **VELOVI**
* **FlatVI**

Future cell states are predicted using **scDiffEq**, and predictions are evaluated against experimentally observed cell populations using **optimal transport (OT)**.

The benchmark evaluates both the latent representations themselves and their downstream performance for future cell-state prediction.

### Main research question

> **Does the choice of latent representation affect future cell-state prediction from single-cell RNA-seq data?**

---

## Methods

### Latent representations

The benchmark compares four different approaches to latent representation learning.

| Method     | Description                                                                  |
| ---------- | ---------------------------------------------------------------------------- |
| **PCA**    | Linear and deterministic dimensionality reduction                            |
| **scVI**   | Variational autoencoder for single-cell RNA-seq data                         |
| **VELOVI** | Probabilistic model incorporating spliced and unspliced RNA and RNA velocity |
| **FlatVI** | scVI-based representation with geometric regularization of the latent space  |

The representations are evaluated at five latent dimensionalities:

**10, 20, 50, 100 and 200 dimensions.**

### Future-state prediction

Future cell states are predicted using **scDiffEq**, a neural stochastic differential equation framework.

The model learns drift and diffusion functions in the latent space and generates multiple stochastic trajectories from an initial cell population.

Predicted populations are compared with experimentally observed populations using optimal transport.

The simulations use the Euler–Maruyama integration scheme.

---

## Dataset

The benchmark uses the **LARRY lineage-tracing dataset**, which provides longitudinal lineage information for single cells.

The dataset contains approximately:

* **49,302 cells**
* **5,864 independent clones**
* **23,420 genes**
* *Mus musculus*
* Three observed time points

LARRY lineage information makes it possible to compare predicted future cell states with experimentally observed descendants.

> **Dataset source:** https://figshare.com/articles/dataset/LARRY_AnnData_feature_inclusive_/29329532



### Train/validation split

The dataset is split at the **clone level** to prevent cells belonging to the same clone from appearing in both training and validation sets.

* **80% of clones:** training
* **20% of clones:** validation

The latent representation models are trained using the complete dataset. The clone-level split is applied to the scDiffEq prediction benchmark.

---

## Data preprocessing

A common preprocessing strategy was applied across the benchmark.

### Highly variable genes

The same set of **2,000 highly variable genes (HVGs)** is used for all representation methods.

### PCA

For PCA, the expression matrix is:

1. normalized to counts per million,
2. scaled to a target library size of \(10^4\),
3. transformed using `log1p`,
4. centered and scaled before PCA.

### scVI, VELOVI and FlatVI

The probabilistic representation models operate directly on the count matrices.

scVI and FlatVI use the spliced expression layer.

VELOVI uses both spliced and unspliced expression layers.

---

## Benchmark configuration

The benchmark explores the effect of both latent representation and neural SDE architecture.

### Representation models

Five latent dimensions are evaluated:

```text
10
20
50
100
200
```

### scDiffEq models

Three hidden-layer dimensions are evaluated:

```text
64
128
256
```

Three independent random seeds are used for each configuration.

This results in:

```text
4 representation methods
× 5 latent dimensions
× 3 hidden dimensions
× 3 random seeds
= 180 scDiffEq models
```

### Simulation parameters

| Parameter                 |                Value |
| ------------------------- |---------------------:|
| Latent dimensions         | 10, 20, 50, 100, 200 |
| Hidden dimensions         |         64, 128, 256 |
| Random seeds              |                    3 |
| Cells simulated per clone |                   32 |
| Integration scheme        |       Euler–Maruyama |
| Integration time step     |                 0.05 |
| Maximum epochs            |                  200 |
| Batch size                |           128 clones |
| Optimizer                 |                 Adam |
| Learning rate             |       10<sup>3</sup> |
| Early stopping            |                  Yes |

Predictions are generated at multiple future time points. The later prediction is obtained by continuing the stochastic simulation from the preceding predicted population.

---

## Evaluation

Two complementary evaluation strategies are used.

### 1. Reconstruction quality

The ability of each latent representation to preserve the observed gene-expression state is evaluated independently from future-state prediction.

For each representation:

1. the original cells are encoded into the latent space;
2. the latent representations are decoded;
3. reconstructed gene-expression profiles are obtained;
4. negative reconstructed values are clipped to zero;
5. each reconstructed cell is normalized to sum to one;
6. reconstructed and original cells are compared using paired distances.

Three distance metrics are used:

* **L2 distance**
* **Cosine distance**
* **Hellinger-style distance**

The reconstruction benchmark therefore evaluates how much information about the current gene-expression state is preserved by each latent representation.

---

### 2. Future cell-state prediction

Future cell-state prediction is evaluated at both the latent and gene-expression levels.

For each validation clone:

1. initial cells are provided to the scDiffEq model;
2. 32 future cells are simulated per clone;
3. predicted populations are compared with the corresponding experimentally observed population;
4. optimal transport is used to quantify the discrepancy.

#### Latent-space evaluation

The predicted and observed cells are compared directly in their corresponding latent representation.

#### Gene-expression-space evaluation

Predicted latent states are decoded into gene-expression profiles.

The resulting predictions are then compared in a common gene-expression space.

This provides a representation-independent evaluation space, although it introduces dependence on the decoder associated with each latent representation.

---

## Optimal transport

Optimal transport is used to compare predicted and observed cell populations.

The main evaluation uses the Euclidean cost in the relevant representation space.

The OT results are normalized using a baseline comparison between the initial and future observed populations:

$$
\text{Normalized OT}
=
\frac{\text{OT}_{\text{prediction, target}}}
{\text{OT}_{\text{initial, target}}}
$$


A normalized value of:

* **1** corresponds to the initial-state baseline;
* **< 1** indicates an improvement over the baseline;
* **> 1** indicates worse performance than the baseline.

For gene-expression-space evaluation, three cost functions are investigated:

* Euclidean L2 distance
* Cosine distance
* Hellinger-style distance

---

## Results

### Reconstruction quality

The reconstruction benchmark shows differences between the four latent representations.

Overall:

* **scVI** provides the lowest L2 and cosine reconstruction errors.
* **PCA** provides the lowest Hellinger-style reconstruction error, with performance improving with latent dimensionality.
* **FlatVI** shows a substantial improvement in cosine reconstruction error as latent dimensionality increases.
* **VELOVI** shows the largest reconstruction distances in the current benchmark.

These results indicate that reconstruction quality depends both on the latent representation and on the distance metric used for evaluation.

---

### Future cell-state prediction in latent space

In latent space, all four representation methods achieve normalized OT values below the initial-state baseline.

The best-performing configurations are:

| Method   |           Best \(t_4\) |           Best \(t_6\) |
| -------- | ---------------------: | ---------------------: |
| PCA      |     0.556 ± 0.012 (20) |     0.529 ± 0.040 (20) |
| **scVI** | **0.491 ± 0.015 (20)** | **0.432 ± 0.009 (10)** |
| VELOVI   |    0.549 ± 0.011 (100) |     0.515 ± 0.013 (50) |
| FlatVI   |     0.534 ± 0.021 (20) |     0.548 ± 0.043 (20) |

scVI provides the lowest normalized OT values at both prediction time points.

However, the differences between representation methods are relatively moderate in latent space.

Because the latent spaces have different geometries, these comparisons should be interpreted with caution.

---

### Future cell-state prediction in gene-expression space

When predictions are decoded into gene-expression space, larger differences between representation methods become apparent.

#### \(t_4\)

| Method |                     L2 |                 Cosine |              Hellinger |
| ------ | ---------------------: | ---------------------: | ---------------------: |
| PCA    | **0.769 ± 0.010 (10)** |     0.861 ± 0.008 (10) |     0.878 ± 0.006 (50) |
| scVI   |    0.823 ± 0.005 (100) | **0.727 ± 0.008 (10)** | **0.865 ± 0.002 (10)** |
| VELOVI |    1.014 ± 0.005 (200) |     0.958 ± 0.002 (10) |     0.990 ± 0.001 (10) |
| FlatVI |     0.972 ± 0.003 (50) |     0.961 ± 0.007 (50) |     0.978 ± 0.001 (20) |

#### \(t_6\)

| Method |                     L2 |                 Cosine |              Hellinger |
| ------ | ---------------------: | ---------------------: | ---------------------: |
| PCA    | **0.692 ± 0.043 (10)** |     0.775 ± 0.005 (10) |     0.808 ± 0.006 (50) |
| scVI   |    0.772 ± 0.014 (100) | **0.624 ± 0.008 (10)** | **0.790 ± 0.009 (10)** |
| VELOVI |     0.992 ± 0.013 (10) |     0.924 ± 0.007 (10) |     0.975 ± 0.002 (10) |
| FlatVI |     0.964 ± 0.005 (50) |     0.926 ± 0.015 (50) |     0.959 ± 0.004 (50) |

PCA and scVI generally provide better future-state prediction in gene-expression space than VELOVI and FlatVI.

The ranking is also dependent on the selected distance metric. PCA performs particularly well according to L2 distance, whereas scVI provides the best cosine and Hellinger-style scores.

---

### Reconstruction and future prediction

The relationship between reconstruction quality and future-state prediction was investigated across the 20 representation configurations.

Both reconstruction error and normalized future-state OT are lower-is-better metrics. Consequently, a positive correlation indicates that representations with lower reconstruction error also tend to achieve lower future-state prediction error.

The strongest global associations are observed for cosine and Hellinger-style distances.

| Metric    | Time    | Pearson \(r\) | Spearman \(\rho\) |
| --------- | ------- | ------------: | ----------------: |
| L2        | \(t_4\) |         0.641 |             0.587 |
| L2        | \(t_6\) |         0.593 |             0.405 |
| Cosine    | \(t_4\) |         0.816 |             0.832 |
| Cosine    | \(t_6\) |         0.873 |             0.859 |
| Hellinger | \(t_4\) |         0.933 |             0.809 |
| Hellinger | \(t_6\) |         0.929 |             0.809 |

These results suggest a strong association between current-state reconstruction quality and future-state prediction performance when comparing different representation families.

However, the relationship is not consistent within individual representation families. This indicates that the global correlation may be partly driven by systematic differences between model families rather than by a universal relationship between reconstruction quality and future prediction.

---

## Repository structure


---

## Installation

Clone the repository:

```bash
git clone https://github.com/Torzzy/scdiffeq_latent_representation_benchmark
cd scdiffeq_reimplementation
```

Create a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```


## Data

The raw LARRY dataset is **not redistributed in this repository**.

Please obtain the dataset from the original data source and place it in the ./data/raw directory.

https://figshare.com/articles/dataset/LARRY_AnnData_feature_inclusive_/29329532

The repository contains the code required to perform preprocessing and downstream analysis once the dataset is available.

---

## Reproducibility

The benchmark is designed to be reproducible from the provided configuration files and scripts.

Random seeds are explicitly controlled for the scDiffEq experiments.

The complete benchmark consists of:

```text
4 representation methods
5 latent dimensions
3 hidden dimensions
3 random seeds
```

for a total of **180 scDiffEq configurations**.

data preprocessing for each methods can be run with:

```bash
python -m script/preprocessing/run_preprocessing.py
```

scdiffeq benchmark can be run with:

```bash
python -m script/benchmark/run_benchmark.py
```

detailed analysis can be run with:

```bash
python -m scdiffeq/analysis/run_analysis.py
```

---

## Computational environment

The experiments were implemented in Python and relied primarily on:

* [PyTorch](https://pytorch.org/)
* [scanPy](https://scanpy.scverse.org/en/stable/)
* [scvi-tools](https://scvi-tools.org/)



The main computational environment used for the experiments was:

```text
Python: 3.11
PyTorch: 2.13.0.dev20260517+cu132
scvi-tools: 1.4.2
scanpy: 1.11.5
```


## Report

The complete scientific report is available in:

```text
pdf/scdiffeq_benchmark.pdf
```



