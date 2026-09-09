import torch

from scvi.module._vae import VAE

from flat_vi_like.flat_decoder import FlatDecoderSCVI


class FlatVAE(VAE):
    """Variational autoencoder used by FlatVI."""

    def __init__(
        self,
        *args,
        lambda_metric: float = 1e-4,
        **kwargs,
    ):
        """Initialize the FlatVI variational autoencoder.

        Parameters
        ----------
        *args
            Positional arguments passed to the parent VAE.
        lambda_metric
            Weight of the metric regularization term.
        **kwargs
            Additional keyword arguments passed to the parent VAE.
        """
        super().__init__(*args, **kwargs)

        self.lambda_metric = lambda_metric
        self.last_metrics = {}

        # Replace the standard scVI decoder with the FlatVI decoder.
        old_decoder = self.decoder

        self.decoder = FlatDecoderSCVI(
            n_input=self.n_latent,
            n_output=self.n_input,
            n_cat_list=old_decoder.px_decoder.n_cat_list,
            n_layers=len(old_decoder.px_decoder.fc_layers),
            inject_covariates=old_decoder.px_decoder.inject_covariates,
            use_batch_norm=False,
            use_layer_norm=False,
        )

        # Initialize the FlatVI decoder from the pretrained scVI decoder.
        self.decoder.load_state_dict(
            old_decoder.state_dict(),
            strict=False,
        )

        self.metric_alpha = torch.nn.Parameter(torch.tensor(10.0))

    def decoder_jacobian(
        self,
        z,
        library,
        batch_index,
    ):
        """Compute the decoder Jacobian with respect to the latent state.

        Parameters
        ----------
        z
            Latent representation with shape ``(N, D)``.
        library
            Log library-size tensor.
        batch_index
            Batch indices or one-hot encoded batch covariates.

        Returns
        -------
        torch.Tensor
            Jacobian of the log-normalized expression with respect to the
            latent representation, with shape ``(N, G, D)``.
        """
        bs, d = z.shape

        eye = (
            torch.eye(d, device=z.device)
            .unsqueeze(0)
            .expand(bs, -1, -1)
            .reshape(-1, d)
        )

        z_rep = (
            z.unsqueeze(1)
            .expand(-1, d, -1)
            .reshape(-1, d)
        )

        library_rep = (
            library.unsqueeze(1)
            .expand(-1, d, -1)
            .reshape(-1, library.shape[-1])
        )

        batch_rep = (
            batch_index.unsqueeze(1)
            .expand(-1, d, -1)
            .reshape(-1, batch_index.shape[-1])
        )

        def decode(latent):
            px_scale, _, _, _ = self.decoder(
                self.dispersion,
                latent,
                library_rep,
                batch_rep,
            )

            return torch.log(px_scale + 1e-8)

        _, jvp = torch.autograd.functional.jvp(
            decode,
            z_rep,
            eye,
            create_graph=True,
        )

        return (
            jvp.view(bs, d, -1)
            .permute(0, 2, 1)
        )

    def metric_regularization(
        self,
        z,
        library,
        batch_index,
    ):
        """Compute the FlatVI metric regularization loss.

        Parameters
        ----------
        z
            Latent representation with shape ``(N, D)``.
        library
            Log library-size tensor.
        batch_index
            Batch indices or one-hot encoded batch covariates.

        Returns
        -------
        torch.Tensor
            Scalar metric regularization loss.
        """

        def check(name, x):
            if torch.is_tensor(x) and not torch.isfinite(x).all():
                print(f"\n[NaN DETECTED] {name}")
                print(f"shape : {tuple(x.shape)}")
                print(f"n_nan : {torch.isnan(x).sum().item()}")
                print(f"n_inf : {torch.isinf(x).sum().item()}")
                print(
                    "min/max :",
                    torch.nan_to_num(x).min().item(),
                    torch.nan_to_num(x).max().item(),
                )

        # Check inputs before applying the decoder.
        check("z", z)
        check("library", library)
        check("batch_index", batch_index)

        px_scale, px_r, px_rate, _ = self.decoder(
            self.dispersion,
            z,
            library,
            batch_index,
        )

        # Check decoder outputs.
        check("px_scale", px_scale)
        check("px_r", px_r)
        check("px_rate", px_rate)

        # Compute the Fisher information term.
        mu = px_rate

        check("mu", mu)

        if self.dispersion == "gene":

            theta = torch.exp(self.px_r)[None].expand_as(mu)

        elif self.dispersion == "gene-batch":

            theta = torch.exp(
                torch.nn.functional.linear(
                    batch_index.float(),
                    self.px_r,
                )
            )

        elif self.dispersion == "gene-cell":

            theta = torch.exp(px_r)

        else:
            raise NotImplementedError

        check("theta", theta)

        fisher = theta / (
            mu * (theta + mu)
            + 1e-8
        )

        check("fisher", fisher)

        # Compute the decoder Jacobian with respect to the latent state.
        J = self.decoder_jacobian(
            z,
            library,
            batch_index,
        )

        check("J", J)

        # Construct the Fisher information metric tensor.
        G = torch.einsum(
            "bij,bik->bjk",
            J,
            fisher.unsqueeze(-1) * J,
        )

        check("G", G)

        # Penalize deviations from the target flat metric.
        check("metric_alpha", self.metric_alpha)

        eye = torch.eye(
            G.size(-1),
            device=G.device,
            dtype=G.dtype,
        )[None]

        target = self.metric_alpha * eye

        out = (
            (G - target)
            .pow(2)
            .sum(dim=(-1, -2))
            .mean()
        )

        check("metric_loss", out)

        return out

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight=1.0,
    ):
        """Compute the VAE loss with FlatVI metric regularization.

        Parameters
        ----------
        tensors
            Input tensors used by the VAE.
        inference_outputs
            Outputs produced by the inference network.
        generative_outputs
            Outputs produced by the generative network.
        kl_weight
            Weight applied to the KL divergence term.

        Returns
        -------
        LossOutput
            VAE loss augmented with the FlatVI metric regularization term.
        """
        losses = super().loss(
            tensors=tensors,
            inference_outputs=inference_outputs,
            generative_outputs=generative_outputs,
            kl_weight=kl_weight,
        )

        metric_loss = self.metric_regularization(
            inference_outputs["z"],
            inference_outputs["library"],
            tensors["batch"],
        )

        losses.loss = (
            losses.loss
            + self.lambda_metric * metric_loss
        )

        recon = losses.reconstruction_loss
        if isinstance(recon, dict):
            recon = recon["reconstruction_loss"]

        self.last_metrics = {
            "reconstruction_loss": recon.mean().item(),
            "kl_loss": losses.kl_local[
                "kl_divergence_z"
            ].mean().item(),
            "metric_loss": metric_loss.item(),
            "total_loss": losses.loss.item(),
        }

        return losses