import torch
import torch.nn as nn


class Diffusion(nn.Module):
    """Neural network predicting the diffusion scale."""

    def __init__(
        self,
        latent_dim,
        hidden=256,
    ):
        """Initialize the diffusion network.

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
        """Compute the positive diffusion scale.

        Parameters
        ----------
        z
            Latent states with shape ``(N, latent_dim)``.
        t
            Time values, either scalar or one value per sample.

        Returns
        -------
        torch.Tensor
            Positive diffusion scale with the same shape as ``z``.

        Raises
        ------
        FloatingPointError
            If the network produces non-finite values.
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

        log_sigma = self.net(
            x
        )

        if not torch.isfinite(
            log_sigma
        ).all():
            bad = ~torch.isfinite(
                log_sigma
            )

            print()
            print(
                "[Diffusion] NON-FINITE log_sigma"
            )
            print(
                "t:",
                t.min().item(),
                t.max().item(),
            )
            print(
                "z max |x|:",
                z.abs().max().item(),
            )
            print(
                "log_sigma finite:",
                torch.isfinite(
                    log_sigma
                ).all().item(),
            )
            print(
                "bad:",
                bad.sum().item(),
            )

            finite = log_sigma[
                torch.isfinite(
                    log_sigma
                )
            ]

            if finite.numel():
                print(
                    "log_sigma finite min:",
                    finite.min().item(),
                )
                print(
                    "log_sigma finite max:",
                    finite.max().item(),
                )

            raise FloatingPointError(
                "Diffusion network produced non-finite log_sigma."
            )

        if log_sigma.max() > 80:
            print()
            print(
                "[Diffusion] EXPLOSIVE log_sigma"
            )
            print(
                "t:",
                t.min().item(),
                t.max().item(),
            )
            print(
                "z max |x|:",
                z.abs().max().item(),
            )
            print(
                "log_sigma min:",
                log_sigma.min().item(),
            )
            print(
                "log_sigma max:",
                log_sigma.max().item(),
            )
            print(
                "sigma max would be approximately:",
                torch.exp(
                    torch.clamp(
                        log_sigma.max(),
                        max=80,
                    )
                ).item(),
            )

        sigma = (
            torch.nn.functional.softplus(
                log_sigma
            )
            + 1e-4
        )

        return sigma