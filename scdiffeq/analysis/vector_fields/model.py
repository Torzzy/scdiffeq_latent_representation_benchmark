import torch

from scdiffeq.models.sde import LatentSDE


def load_vector_field_model(
    checkpoint_path,
    latent_dim,
    hidden_dim,
    device="cuda",
):
    model = LatentSDE(
        latent_dim=latent_dim,
        hidden=hidden_dim,
    ).to(device)

    state = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        isinstance(state, dict)
        and "model_state_dict" in state
    ):
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.eval()

    return model