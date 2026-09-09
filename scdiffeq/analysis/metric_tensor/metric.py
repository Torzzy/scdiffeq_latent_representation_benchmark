import torch

from .jacobian import (
    decoder_jacobian,
)


def metric_tensor(
    model,
    z,
    library,
    batch_index,
):
    """Compute the Fisher metric tensor of the decoder.

    Parameters
    ----------
    model
        Model containing the decoder and dispersion parameters.
    z
        Latent representation with shape ``(N, D)``.
    library
        Log library-size tensor.
    batch_index
        Batch indices or one-hot encoded batch covariates.

    Returns
    -------
    torch.Tensor
        Fisher metric tensor with shape ``(N, D, D)``.
    """
    _, px_r, px_rate, _ = model.decoder(
        model.dispersion,
        z,
        library,
        batch_index,
    )

    mu = px_rate

    if model.dispersion == "gene":
        theta = torch.exp(
            model.px_r,
        )[None].expand_as(mu)

    elif model.dispersion == "gene-batch":
        theta = torch.exp(
            torch.nn.functional.linear(
                batch_index.float(),
                model.px_r,
            )
        )

    elif model.dispersion == "gene-cell":
        theta = torch.exp(
            px_r,
        )

    else:
        raise NotImplementedError

    fisher = theta / (
        mu * (theta + mu)
        + 1e-8
    )

    J = decoder_jacobian(
        model,
        z,
        library,
        batch_index,
    )

    G = torch.einsum(
        "bij,bik->bjk",
        J,
        fisher.unsqueeze(-1) * J,
    )

    return G