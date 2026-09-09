import torch.nn as nn

from scdiffeq.models.diffusion import Diffusion
from scdiffeq.models.drift import Drift


class LatentSDE(nn.Module):
    """Latent stochastic differential equation model."""

    def __init__(
        self,
        latent_dim,
        hidden=256,
    ):
        """Initialize the latent SDE model.

        Parameters
        ----------
        latent_dim
            Dimensionality of the latent representation.
        hidden
            Number of hidden units in the drift and diffusion networks.
        """
        super().__init__()

        self.drift = Drift(
            latent_dim,
            hidden,
        )

        self.diffusion = Diffusion(
            latent_dim,
            hidden,
        )

    def f(
        self,
        t,
        z,
    ):
        """Compute the drift function.

        Parameters
        ----------
        t
            Time values.
        z
            Latent states.

        Returns
        -------
        torch.Tensor
            Drift values for the given latent states and times.
        """
        return self.drift(
            z,
            t,
        )

    def g(
        self,
        t,
        z,
    ):
        """Compute the diffusion function.

        Parameters
        ----------
        t
            Time values.
        z
            Latent states.

        Returns
        -------
        torch.Tensor
            Diffusion values for the given latent states and times.
        """
        return self.diffusion(
            z,
            t,
        )