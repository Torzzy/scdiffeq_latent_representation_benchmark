
import torch
import torch.nn.functional as F

from scvi.nn import DecoderSCVI


class FlatDecoderSCVI(DecoderSCVI):
    """
    Decoder used by FlatVI to map latent representations to
    gene-expression parameters.

    The decoder follows the SCVI parameterization for the
    normalized expression proportions and library-scaled
    negative-binomial rate. It additionally supports the
    gene-cell dispersion parameterization inherited from SCVI.

    Notes
    -----
    The decoder returns the same four quantities as the standard
    SCVI decoder:

    - normalized expression proportions
    - dispersion parameters
    - library-scaled expression rates
    - dropout logits
    """

    def forward(
        self,
        dispersion,
        z,
        library,
        *cat_list,
    ):
        """
        Decode latent representations into SCVI expression parameters.

        Parameters
        ----------
        dispersion
            Dispersion parameterization used by the model.
        z
            Latent representation with shape ``(n_cells, n_latent)``.
        library
            Log library size for each cell. If ``None``, the decoder
            returns normalized expression proportions as the rate.
        *cat_list
            Additional categorical covariates passed to the SCVI
            decoder.

        Returns
        -------
        tuple
            A tuple containing:

            ``px_scale``
                Normalized gene-expression proportions.

            ``px_r``
                Gene-cell dispersion parameters when
                ``dispersion == "gene-cell"``, otherwise ``None``.

            ``px_rate``
                Library-scaled expression rate used by the
                negative-binomial likelihood.

            ``px_dropout``
                Dropout logits produced by the decoder.
        """

        # Decode the latent representation into the hidden representation
        # used by the SCVI expression heads.
        h = self.px_decoder(
            z,
            *cat_list,
        )

        # Convert decoder outputs into normalized gene-expression proportions.
        px_scale = F.softmax(
            self.px_scale_decoder[0](h),
            dim=-1,
        )

        # Scale the normalized expression by the cell-specific library size.
        if library is None:
            px_rate = px_scale
        else:
            px_rate = torch.exp(library) * px_scale

        # Gene-cell dispersion requires a cell-specific dispersion value.
        if dispersion == "gene-cell":
            px_r = self.px_r_decoder(h)
        else:
            px_r = None

        # Predict dropout logits for the zero-inflated likelihood parameterization.
        px_dropout = self.px_dropout_decoder(h)

        return (
            px_scale,
            px_r,
            px_rate,
            px_dropout,
        )

