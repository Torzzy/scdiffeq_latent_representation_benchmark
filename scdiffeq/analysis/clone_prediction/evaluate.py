import torch

from scdiffeq.analysis.clone_prediction.model import (
    load_clone_prediction_model,
)
from scdiffeq.analysis.clone_prediction.plotting import (
    plot_clone_prediction,
)
from scdiffeq.analysis.clone_prediction.projection import (
    project_clone_prediction,
)
from scdiffeq.simulations.euler_maruyama import (
    EulerMaruyamaSimulator,
)


@torch.no_grad()
def evaluate_clone_prediction(
    adata,
    loader,
    checkpoint_path,
    output_dir,
    latent_dim,
    hidden_dim,
    seed,
    clone_id=0,
    latent_key="X_latent",
    dt=0.05,
    n_simulations=200,
    device="cuda",
):
    """Evaluate and plot a clone prediction from a trained latent SDE.

    Parameters
    ----------
    adata
        Annotated data matrix containing the latent representation and
        associated observations.
    loader
        DataLoader containing initial states and future clone populations.
    checkpoint_path
        Path to the trained model checkpoint.
    output_dir
        Directory where the prediction plot will be saved.
    latent_dim
        Dimensionality of the latent representation.
    hidden_dim
        Number of hidden units used by the model.
    seed
        Random seed associated with the model configuration.
    clone_id
        Global index of the clone to evaluate.
    latent_key
        Key in ``adata.obsm`` containing the latent representation.
    dt
        Time step used by the Euler-Maruyama simulator.
    n_simulations
        Number of simulated trajectories used to generate the predicted
        population.
    device
        Device on which the model and simulations are run.
    """
    model = load_clone_prediction_model(
        checkpoint_path,
        latent_dim,
        hidden_dim,
        device=device,
    )

    simulator = EulerMaruyamaSimulator(
        model,
        dt=dt,
    )

    batch = None
    local_idx = None
    running = 0

    for b in loader:
        B = len(
            b["future_sets"],
        )

        if clone_id < running + B:
            batch = b
            local_idx = clone_id - running
            break

        running += B

    if batch is None:
        raise ValueError(
            "clone_id outside loader."
        )

    z0 = batch["z0"][
        local_idx:local_idx + 1
    ].to(device)

    gt_t1 = batch["future_sets"][
        local_idx
    ][4].cpu().numpy()

    gt_t2 = batch["future_sets"][
        local_idx
    ][6].cpu().numpy()

    pred_t1 = simulator.sample_population(
        z0,
        2,
        4,
        n_simulations,
    ).squeeze(0)

    pred_t2 = simulator.sample_population(
        z0,
        2,
        6,
        n_simulations,
    ).squeeze(0)

    projections = project_clone_prediction(
        adata=adata,
        latent_key=latent_key,
        z0=z0.cpu().numpy(),
        gt_t1=gt_t1,
        gt_t2=gt_t2,
        pred_t1=pred_t1.cpu().numpy(),
        pred_t2=pred_t2.cpu().numpy(),
    )

    plot_clone_prediction(
        projections=projections,
        output_path=(
            output_dir
            / f"clone_{clone_id}.png"
        ),
        clone_id=clone_id,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        seed=seed,
    )

    model.cpu()

    del model
    del simulator

    torch.cuda.empty_cache()


if __name__ == "__main__":
    import scanpy as sc
    from pathlib import Path

    adata = sc.read_h5ad(
        "../../../data/processed/larry_flatvi_20.h5ad"
    )

    out_dir = "../../../manifold_results/"
    out_dir = Path(out_dir)

    ckpt_path = "../../../manifold_results/best_model.pt"

    from scdiffeq.data.dataloader import build_dataloaders
    from scdiffeq.data.pair_dict import build_pair_dict

    pair_dict = build_pair_dict(adata)

    train_loader, val_loader = build_dataloaders(
        adata,
        pair_dict,
        latent_key="X_latent",
    )

    hidden_dim = 256
    latent_dim = 20
    seed = 0

    evaluate_clone_prediction(
        adata=adata,
        loader=val_loader,
        checkpoint_path=ckpt_path,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        seed=seed,
        output_dir=out_dir,
        clone_id=50,
        n_simulations=32,
    )