import torch

from scdiffeq.models.sde import LatentSDE


def load_clone_prediction_model(
    checkpoint_path,
    latent_dim,
    hidden_dim,
    device="cuda",
):
    """Load a trained latent SDE model for clone prediction.

    Parameters
    ----------
    checkpoint_path
        Path to the trained model checkpoint.
    latent_dim
        Dimensionality of the latent representation.
    hidden_dim
        Number of hidden units in the model.
    device
        Device on which the model is loaded.

    Returns
    -------
    LatentSDE
        Loaded latent SDE model in evaluation mode.
    """
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

    model.load_state_dict(
        state,
    )

    model.eval()

    return model