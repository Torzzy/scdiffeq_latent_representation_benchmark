from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from config import RESULTS_DIR

def plot():
    RESULTS_FILE = RESULTS_DIR / "reconstruction_metrics.csv"


    OUTPUT_FILE = RESULTS_DIR / "reconstruction_quality.png"



    METHOD_ORDER = [
        "PCA",
        "SCVI",
        "VELOVI",
        "FlatVI",
    ]

    METRICS = [
        (
            "l2_mean",
            "L2 distance",
        ),
        (
            "cosine_mean",
            "Cosine distance",
        ),
        (
            "hellinger_mean",
            "Hellinger-style distance",
        ),
    ]


    # ============================================================
    # Load data
    # ============================================================

    df = pd.read_csv(
        RESULTS_FILE
    )

    df["method"] = pd.Categorical(
        df["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    df = df.sort_values(
        [
            "method",
            "latent_dim",
        ]
    )


    # ============================================================
    # Plot
    # ============================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
        sharex=True,
    )

    for ax, (metric, ylabel) in zip(
        axes,
        METRICS,
    ):

        for method in METHOD_ORDER:

            subset = df[
                df["method"] == method
            ]

            ax.plot(
                subset["latent_dim"],
                subset[metric],
                marker="o",
                linewidth=2,
                label=method,
            )

        ax.set_xlabel(
            "Latent dimensionality"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_xticks(
            [10, 20, 50, 100, 200]
        )

        ax.grid(
            True,
            alpha=0.3,
        )


    axes[0].set_title(
        "Euclidean L2 reconstruction"
    )

    axes[1].set_title(
        "Cosine reconstruction"
    )

    axes[2].set_title(
        "Hellinger reconstruction"
    )

    axes[-1].legend(
        frameon=False,
        loc="best",
    )


    fig.suptitle(
        "Reconstruction quality across latent representations",
        fontsize=14,
    )


    fig.tight_layout()


    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


    print(
        f"Figure saved to: {OUTPUT_FILE}"
    )

if __name__ == "__main__":
    plot()