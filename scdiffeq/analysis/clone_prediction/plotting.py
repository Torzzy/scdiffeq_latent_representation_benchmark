from pathlib import Path

import matplotlib.pyplot as plt


def plot_clone_prediction(
    projections,
    output_path,
    clone_id,
    latent_dim,
    hidden_dim,
    seed,
):
    """Plot the predicted and observed populations for one clone.

    Parameters
    ----------
    projections
        Dictionary containing the latent representation, initial state,
        ground-truth populations, and simulated populations.
    output_path
        Path where the generated figure will be saved.
    clone_id
        Identifier of the clone being evaluated.
    latent_dim
        Dimensionality of the latent representation.
    hidden_dim
        Number of hidden units used by the model.
    seed
        Random seed associated with the model configuration.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    fig.suptitle(
        (
            f"Clone {clone_id}\n"
            f"latent={latent_dim}   "
            f"hidden={hidden_dim}   "
            f"seed={seed}"
        ),
        fontsize=16,
    )

    panels = [
        (
            projections["pred_t1"],
            projections["gt_t1"],
            "t = 4",
        ),
        (
            projections["pred_t2"],
            projections["gt_t2"],
            "t = 6",
        ),
    ]

    for axis, (pred, gt, title) in zip(
        ax,
        panels,
    ):
        axis.scatter(
            projections["latent"][:, 0],
            projections["latent"][:, 1],
            s=2,
            color="lightgrey",
            alpha=0.25,
        )

        axis.scatter(
            gt[:, 0],
            gt[:, 1],
            color="tab:blue",
            s=25,
            label="Ground truth",
        )

        axis.scatter(
            pred[:, 0],
            pred[:, 1],
            color="tab:red",
            s=12,
            alpha=0.35,
            label="Simulation",
        )

        axis.scatter(
            projections["z0"][:, 0],
            projections["z0"][:, 1],
            color="black",
            marker="*",
            s=120,
            label="Initial cell",
        )

        axis.set_title(
            f"{title}\n{len(pred)} simulated cells"
        )

        axis.set_xticks([])
        axis.set_yticks([])

        axis.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)