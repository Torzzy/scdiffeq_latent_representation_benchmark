import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_metric(
    z,
    values,
    output_path,
    title="",
    log_value=True,
):
    """Plot a scalar metric on a PCA projection of the latent space.

    Parameters
    ----------
    z
        Latent representation used to compute the two-dimensional PCA
        projection.
    values
        Scalar metric values associated with each latent point.
    output_path
        Path where the generated figure will be saved.
    title
        Title displayed on the figure.
    log_value
        Whether to apply a base-10 logarithmic transformation to the
        metric values before plotting.
    """
    embedding = PCA(
        n_components=2,
    ).fit_transform(
        z,
    )

    if log_value:
        values = np.log10(
            values + 1e-12,
        )

    plt.figure(
        figsize=(6, 5),
    )

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=values,
        cmap="viridis",
        s=8,
    )

    plt.colorbar()

    plt.title(
        title,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
    )

    plt.close()