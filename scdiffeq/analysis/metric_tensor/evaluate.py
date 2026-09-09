import matplotlib.pyplot as plt
import numpy as np

from .alpha import (
    get_metric_alpha,
)
from .compute import (
    compute_metric_tensor,
)
from .statistics import (
    compute_metric_statistics,
    cleanup_metric_tensor,
)
from .plotting import (
    plot_metric,
)


def evaluate_metric_tensor(
    adata,
    model,
    output_dir,
    n_cells=20_000,
):
    """Run the complete metric tensor analysis.

    Parameters
    ----------
    adata
        Annotated data matrix containing the latent representation.
    model
        Trained model used to compute the metric tensor.
    output_dir
        Directory where analysis figures are saved.
    n_cells
        Maximum number of cells used for the metric computation.

    Returns
    -------
    dict
        Summary statistics describing the learned metric tensor.
    """
    (
        G,
        z,
        neighbors,
        memmap_path,
    ) = compute_metric_tensor(
        model=model,
        adata=adata,
        n_cells=n_cells,
    )

    stats = compute_metric_statistics(
        G,
        neighbors,
    )

    alpha = get_metric_alpha(
        model,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_metric(
        z,
        stats["G_norm"],
        output_dir / "norm_G.png",
        title="||G||",
    )

    plot_metric(
        z,
        stats["conditioning"],
        output_dir / "conditioning.png",
        title="conditioning",
        log_value=False,
    )

    plot_metric(
        z,
        stats["eig_min"],
        output_dir / "eig_min.png",
        title="eig_min",
        log_value=False,
    )

    plot_metric(
        z,
        stats["eig_max"],
        output_dir / "eig_max.png",
        title="eig_max",
        log_value=False,
    )

    eigvals = np.sort(
        stats["eigvals"],
        axis=1,
    )[:, ::-1]

    mean_eigvals = eigvals.mean(
        axis=0,
    )

    fig, ax = plt.subplots(
        figsize=(5, 4),
    )

    ax.plot(
        np.arange(
            1,
            len(mean_eigvals) + 1,
        ),
        mean_eigvals,
        lw=2,
    )

    ax.set_yscale(
        "log",
    )

    ax.set_xlabel(
        "Eigenvalue rank",
    )

    ax.set_ylabel(
        "Mean eigenvalue",
    )

    ax.set_title(
        "Metric tensor spectrum",
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "eigenvalue_spectrum.png",
        dpi=300,
    )

    plt.close(
        fig,
    )

    summary = {
        "alpha": alpha,
        "metric_variation_mean":
            stats["metric_variation_mean"],
        "conditioning_mean":
            np.mean(
                stats["conditioning"],
            ),
        "conditioning_median":
            np.median(
                stats["conditioning"],
            ),
        "conditioning_max":
            np.max(
                stats["conditioning"],
            ),
        "effective_conditioning_mean":
            stats["effective_conditioning_mean"],
        "effective_conditioning_std":
            stats["effective_conditioning_std"],
        "energy_rank_mean":
            stats["energy_rank_mean"],
        "energy_rank_std":
            stats["energy_rank_std"],
        "eig_min_mean":
            np.mean(
                stats["eig_min"],
            ),
        "eig_max_mean":
            np.mean(
                stats["eig_max"],
            ),
    }

    cleanup_metric_tensor(
        G,
        memmap_path,
    )

    return summary