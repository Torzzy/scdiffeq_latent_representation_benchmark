from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt


MARKERS = {
    "pca": "o",
    "scvi": "^",
    "velovi": "s",
    "flatvi": "p",
}

CMAP = plt.cm.viridis


def best_models(
    summary_df,
):
    """Select the best model for each method and latent dimension.

    The best model is defined as the one with the lowest normalized OT
    at the final evaluation time.

    Parameters
    ----------
    summary_df
        DataFrame containing benchmark summary metrics.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the best model for each
        ``(method, latent_dim)`` combination.
    """
    idx = (
        summary_df
        .groupby(
            [
                "method",
                "latent_dim",
            ]
        )["normalized_ot_t2_mean"]
        .idxmin()
    )

    return summary_df.loc[idx].copy()


def plot_tradeoff(
    summary_df,
    output_path,
):
    """Plot the drift-diffusion tradeoff for the best models.

    Parameters
    ----------
    summary_df
        DataFrame containing benchmark summary metrics.
    output_path
        Path where the generated figure will be saved.
    """
    df = best_models(
        summary_df,
    )

    latent_dims = sorted(
        df.latent_dim.unique(),
    )

    colors = {
        d: CMAP(
            i / max(
                len(latent_dims) - 1,
                1,
            )
        )
        for i, d in enumerate(
            latent_dims,
        )
    }

    fig, ax = plt.subplots(
        figsize=(7, 6),
    )

    for _, row in df.iterrows():
        ax.scatter(
            row["ratio_mean_mean"],
            row["normalized_ot_t2_mean"],
            marker=MARKERS[row.method],
            color=colors[row.latent_dim],
            s=150,
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_xlabel(
        "Mean drift / diffusion ratio",
    )

    ax.set_ylabel(
        "Relative OT",
    )

    ax.grid(
        alpha=0.3,
    )

    marker_handles = [
        mlines.Line2D(
            [],
            [],
            color="black",
            marker=m,
            linestyle="None",
            markersize=8,
            label=name.upper(),
        )
        for name, m in MARKERS.items()
    ]

    color_handles = [
        mlines.Line2D(
            [],
            [],
            color=c,
            marker="o",
            linestyle="None",
            markersize=8,
            label=str(d),
        )
        for d, c in colors.items()
    ]

    legend = ax.legend(
        handles=marker_handles,
        title="Representation",
        loc="upper left",
    )

    ax.add_artist(
        legend,
    )

    ax.legend(
        handles=color_handles,
        title="Latent dim",
        loc="lower right",
    )

    fig.tight_layout()

    output_path = Path(
        output_path,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
    )

    plt.close(
        fig,
    )


def make_plots(
    summary_df,
    output_dir,
):
    """Generate all benchmark plots.

    Parameters
    ----------
    summary_df
        DataFrame containing benchmark summary metrics.
    output_dir
        Directory where generated figures will be saved.
    """
    output_dir = Path(
        output_dir,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_tradeoff(
        summary_df,
        output_dir / "tradeoff.png",
    )