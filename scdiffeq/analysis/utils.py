from pathlib import Path

import scvi
import torch

from flat_vi_like.flat_model import (
    FlatModel,
)


GROUP_COLUMNS = [
    "method",
    "latent_dim",
    "hidden_dim",
]


def ensure_dir(path):
    """Create a directory if needed.

    Parameters
    ----------
    path
        Path to the directory.

    Returns
    -------
    pathlib.Path
        Path object corresponding to the created directory.
    """
    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def metric_mean(metric):
    """Return the column name corresponding to a metric mean.

    Parameters
    ----------
    metric
        Name of the metric.

    Returns
    -------
    str
        Column name for the metric mean.
    """
    return f"{metric}_mean"


def metric_std(metric):
    """Return the column name corresponding to a metric standard deviation.

    Parameters
    ----------
    metric
        Name of the metric.

    Returns
    -------
    str
        Column name for the metric standard deviation.
    """
    return f"{metric}_std"


def available_metrics(summary_df):
    """Return all metrics available in a summary dataframe.

    Parameters
    ----------
    summary_df
        Dataframe containing benchmark metrics.

    Returns
    -------
    list[str]
        Sorted metric names with corresponding mean columns.
    """
    metrics = []

    for column in summary_df.columns:
        if column.endswith("_mean"):
            metrics.append(
                column[:-5]
            )

    return sorted(
        metrics
    )


def lower_is_better(metric):
    """Determine whether lower values indicate better performance.

    Parameters
    ----------
    metric
        Name of the metric.

    Returns
    -------
    bool
        Whether the metric should be minimized.
    """
    return metric.startswith("ot") or (
        "distance" in metric
    )


def higher_is_better(metric):
    """Determine whether higher values indicate better performance.

    Parameters
    ----------
    metric
        Name of the metric.

    Returns
    -------
    bool
        Whether the metric should be maximized.
    """
    return not lower_is_better(
        metric
    )


def metric_columns(summary_df):
    """Return columns corresponding to benchmark metrics.

    Parameters
    ----------
    summary_df
        Dataframe containing benchmark results.

    Returns
    -------
    list[str]
        Names of columns that are not grouping columns.
    """
    return [
        column
        for column in summary_df.columns
        if column not in GROUP_COLUMNS
    ]


def filter_dimension(
    df,
    latent_dim=None,
    hidden_dim=None,
):
    """Filter a benchmark dataframe by model dimensions.

    Parameters
    ----------
    df
        Benchmark dataframe.
    latent_dim
        Latent dimensionality used for filtering.
    hidden_dim
        Hidden dimensionality used for filtering.

    Returns
    -------
    pandas.DataFrame
        Filtered benchmark dataframe.
    """
    out = df

    if latent_dim is not None:
        out = out[
            out["latent_dim"]
            == latent_dim
        ]

    if hidden_dim is not None:
        out = out[
            out["hidden_dim"]
            == hidden_dim
        ]

    return out


def sort_methods(df):
    """Sort a benchmark dataframe alphabetically by method.

    Parameters
    ----------
    df
        Benchmark dataframe.

    Returns
    -------
    pandas.DataFrame
        Sorted dataframe with a reset index.
    """
    return df.sort_values(
        "method"
    ).reset_index(
        drop=True
    )


def load_vae_model(
    adata,
    latent_dim,
    method,
    processed_dir,
):
    """Load a trained VAE model from a checkpoint.

    Parameters
    ----------
    adata
        Annotated data matrix used to initialize the model.
    latent_dim
        Dimensionality of the latent representation.
    method
        Model type. ``"flatvi"`` loads a FlatVI model; other methods
        load a standard scVI model.
    processed_dir
        Directory containing the trained model checkpoints.

    Returns
    -------
    scvi.model.SCVI
        Loaded VAE model with checkpoint weights.
    """
    if method == "flatvi":
        FlatModel.setup_anndata(
            adata,
        )

        model = FlatModel(
            adata,
            n_latent=latent_dim,
        )

        ckpt = torch.load(
            processed_dir
            / "models"
            / f"flatvi_{latent_dim}"
            / "model.pt",
            map_location="cuda",
            weights_only=False,
        )

        if "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]

        ckpt.pop(
            "pyro_param_store",
            None,
        )

        model.module.load_state_dict(
            ckpt,
            strict=True,
        )

        return model

    scvi.model.SCVI.setup_anndata(
        adata,
    )

    model = scvi.model.SCVI(
        adata,
        n_latent=latent_dim,
    )

    ckpt = torch.load(
        processed_dir
        / "models"
        / f"scvi_{latent_dim}"
        / "model.pt",
        map_location="cuda",
        weights_only=False,
    )

    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]

    ckpt.pop(
        "pyro_param_store",
        None,
    )

    model.module.load_state_dict(
        ckpt,
        strict=True,
    )

    return model