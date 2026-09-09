from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/home/tom/Documents/projets/scdiffeq_reimplementation"
)

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

MODELS_DIR = (
    DATASET_DIR
    / "models"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "reconstruction_evaluation"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "reconstruction_metrics.csv"
)


# ============================================================
# BENCHMARK
# ============================================================

METHODS = [
    "pca",
    "scvi",
    "velovi",
    "flatvi",
]

LATENT_DIMS = [
    10,
    20,
    50,
    100,
    200,
]

BATCH_SIZE = 256


# ============================================================
# DATA
# ============================================================

LAYER = "spliced"

TARGET_SUM = 1e4

EPS = 1e-8

# Seed used for stochastic reconstruction.
# In particular, this makes VELOVI reproducible.
RECONSTRUCTION_SEED = 0