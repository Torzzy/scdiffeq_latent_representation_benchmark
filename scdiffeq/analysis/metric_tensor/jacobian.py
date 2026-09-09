import torch


def decoder_jacobian(
    model,
    z,
    library,
    batch_index,
):
    """Compute the Jacobian of the decoder log-expression.

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
        Decoder Jacobian with shape ``(N, G, D)``, where ``G`` is the
        number of genes and ``D`` is the latent dimensionality.
    """
    batch_size, latent_dim = z.shape

    eye = (
        torch.eye(
            latent_dim,
            device=z.device,
        )
        .unsqueeze(0)
        .expand(
            batch_size,
            -1,
            -1,
        )
        .reshape(
            -1,
            latent_dim,
        )
    )

    z_rep = (
        z.unsqueeze(1)
        .expand(
            -1,
            latent_dim,
            -1,
        )
        .reshape(
            -1,
            latent_dim,
        )
    )

    library_rep = (
        library.unsqueeze(1)
        .expand(
            -1,
            latent_dim,
            -1,
        )
        .reshape(
            -1,
            library.shape[-1],
        )
    )

    batch_rep = (
        batch_index.unsqueeze(1)
        .expand(
            -1,
            latent_dim,
            -1,
        )
        .reshape(
            -1,
            batch_index.shape[-1],
        )
    )

    def decode_log_expression(latent):
        px_scale, _, _, _ = model.decoder(
            model.dispersion,
            latent,
            library_rep,
            batch_rep,
        )

        return torch.log(
            px_scale + 1e-8,
        )

    _, jvp = torch.autograd.functional.jvp(
        decode_log_expression,
        z_rep,
        eye,
        create_graph=False,
    )

    return (
        jvp.view(
            batch_size,
            latent_dim,
            -1,
        )
        .permute(
            0,
            2,
            1,
        )
    )