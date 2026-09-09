from pathlib import Path

import matplotlib.pyplot as plt


MARKERS = {
    "pca": "o",
    "scvi": "^",
    "velovi": "s",
    "flatvi": "p",
}


def plot_normalized_gene_ot(
    summary_df,
    output_path,
    metric,
    ylabel=None,
):
    """Plot mean normalized gene-space OT versus latent dimension.

    Each point corresponds to one model configuration, defined by
    ``(method, latent_dim, hidden_dim)``. The plotted value is the mean
    across the available seeds.

    Standard deviations are retained in the summary dataframe but are
    intentionally not displayed in the figure.

    Parameters
    ----------
    summary_df
        Dataframe containing aggregated benchmark metrics.
    output_path
        Path where the generated figure will be saved.
    metric
        Base name of the normalized OT metric.
    ylabel
        Label for the y-axis. If ``None``, a default label is used.

    Raises
    ------
    KeyError
        If the requested metric column is missing from ``summary_df``.
    """
    mean_column = f"{metric}_mean"

    if mean_column not in summary_df.columns:
        raise KeyError(
            f"Missing column '{mean_column}' "
            f"in summary dataframe."
        )

    fig, ax = plt.subplots(
        figsize=(7, 6),
    )

    for method, df_method in summary_df.groupby(
        "method"
    ):
        for hidden_dim, df in df_method.groupby(
            "hidden_dim"
        ):
            df = df.sort_values(
                "latent_dim"
            )

            label = method.upper()

            if summary_df["hidden_dim"].nunique() > 1:
                label = (
                    f"{method.upper()} "
                    f"(hidden={hidden_dim})"
                )

            ax.plot(
                df["latent_dim"],
                df[mean_column],
                marker=MARKERS.get(
                    method,
                    "o",
                ),
                linestyle="-",
                label=label,
            )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Latent dimension"
    )

    if ylabel is None:
        ylabel = "Relative gene-space OT"

    ax.set_ylabel(
        ylabel
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_normalized_gene_ot_l2(
    summary_df,
    output_path,
):
    """Plot normalized gene-space OT using Euclidean distance.

    Parameters
    ----------
    summary_df
        Dataframe containing aggregated benchmark metrics.
    output_path
        Path where the generated figure will be saved.
    """
    plot_normalized_gene_ot(
        summary_df=summary_df,
        output_path=output_path,
        metric="normalized_gene_ot_l2_t2",
        ylabel="Relative gene-space OT — L2",
    )


def plot_normalized_gene_ot_cosine(
    summary_df,
    output_path,
):
    """Plot normalized gene-space OT using cosine distance.

    Parameters
    ----------
    summary_df
        Dataframe containing aggregated benchmark metrics.
    output_path
        Path where the generated figure will be saved.
    """
    plot_normalized_gene_ot(
        summary_df=summary_df,
        output_path=output_path,
        metric="normalized_gene_ot_cosine_t2",
        ylabel="Relative gene-space OT — cosine",
    )


def plot_normalized_gene_ot_hellinger(
    summary_df,
    output_path,
):
    """Plot normalized gene-space OT using Hellinger distance.

    Parameters
    ----------
    summary_df
        Dataframe containing aggregated benchmark metrics.
    output_path
        Path where the generated figure will be saved.
    """
    plot_normalized_gene_ot(
        summary_df=summary_df,
        output_path=output_path,
        metric="normalized_gene_ot_hellinger_t2",
        ylabel="Relative gene-space OT — Hellinger",
    )


def make_plots(
    summary_df,
    output_dir,
):
    """Generate normalized gene-space OT plots for all distances.

    Parameters
    ----------
    summary_df
        Dataframe containing aggregated benchmark metrics.
    output_dir
        Directory where the generated figures will be saved.
    """
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_normalized_gene_ot_l2(
        summary_df,
        output_dir / "normalized_gene_ot_l2.png",
    )

    plot_normalized_gene_ot_cosine(
        summary_df,
        output_dir / "normalized_gene_ot_cosine.png",
    )

    plot_normalized_gene_ot_hellinger(
        summary_df,
        output_dir / "normalized_gene_ot_hellinger.png",
    )