import torch


def latent_statistics(
    z,
):

    return {

        "variance": z.var().item(),

        "norm": torch.norm(
            z,
            dim=1,
        ).mean().item(),

        "dimension_variance": z.var(
            dim=0,
        ).mean().item(),

    }