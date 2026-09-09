import torch
import torch.nn as nn


class Drift(nn.Module):
    """Neural network predicting the latent drift."""

    def __init__(
        self,
        latent_dim,
        hidden=256,
    ):
        """Initialize the drift network.

        Parameters
        ----------
        latent_dim
            Dimensionality of the latent representation.
        hidden
            Number of hidden units in the network.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                latent_dim + 1,
                hidden,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden,
                hidden,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden,
                latent_dim,
            ),
        )

    def forward(
        self,
        z,
        t,
    ):
        """Compute the latent drift.

        Parameters
        ----------
        z
            Latent states with shape ``(N, latent_dim)``.
        t
            Time values, either scalar or one value per sample.

        Returns
        -------
        torch.Tensor
            Drift values with the same shape as ``z``.
        """
        if not torch.is_tensor(
            t
        ):
            t = torch.tensor(
                t,
                device=z.device,
                dtype=z.dtype,
            )

        if t.ndim == 0:
            t = t.expand(
                len(z)
            )

        t = t[:, None]

        x = torch.cat(
            [
                z,
                t,
            ],
            dim=1,
        )

        return self.net(
            x
        )