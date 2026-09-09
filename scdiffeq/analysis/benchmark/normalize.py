from pathlib import Path

import pandas as pd
import scanpy as sc
import torch

from scdiffeq.benchmark.metrics.ot_distance import ot_distance
from scdiffeq.data.dataloader import build_dataloaders
from scdiffeq.data.pair_dict import build_pair_dict
from scdiffeq.models.sde import LatentSDE
from scdiffeq.simulations.euler_maruyama import EulerMaruyamaSimulator


def safe_mean(values):
    """Compute the mean of a sequence, returning NaN for empty inputs.

    Parameters
    ----------
    values
        Sequence of numeric values.

    Returns
    -------
    float
        Mean value, or NaN if the input is empty.
    """
    if len(values) == 0:
        return float("nan")

    return float(torch.tensor(values).mean())


@torch.no_grad()
def evaluate_normalized_ot(
    model,
    loader,
    n_simulations=128,
):
    """Evaluate normalized OT distances at future time points.

    Parameters
    ----------
    model
        Trained latent SDE model used to simulate future states.
    loader
        DataLoader providing initial latent states and target populations.
    n_simulations
        Number of simulated samples generated for each initial state.

    Returns
    -------
    dict
        Normalized OT scores at times t1 and t2.
    """
    simulator = EulerMaruyamaSimulator(model)

    device = next(model.parameters()).device

    score_t1 = []
    score_t2 = []

    skipped_t1 = 0
    skipped_t2 = 0

    for batch in loader:
        z0 = batch["z0"].to(device)

        future_sets = batch["future_sets"]

        for i in range(len(future_sets)):
            z = z0[i:i + 1]

            target_t1 = future_sets[i][4].to(device)
            target_t2 = future_sets[i][6].to(device)

            pred_t1 = simulator.sample_population(
                z,
                2,
                4,
                n_simulations,
            ).squeeze(0)

            pred_t2 = simulator.continue_simulation(
                pred_t1.unsqueeze(0),
                4,
                6,
            ).squeeze(0)

            # Use the initial latent state as the baseline population.
            z_pop = z.expand(
                n_simulations,
                -1,
            )

            if (
                pred_t1.shape[0] >= 2
                and target_t1.shape[0] >= 2
            ):
                model_ot = ot_distance(
                    pred_t1,
                    target_t1,
                )

                baseline_ot = ot_distance(
                    z_pop,
                    target_t1,
                )

                score_t1.append(
                    model_ot / (baseline_ot + 1e-8)
                )
            else:
                skipped_t1 += 1

            if (
                pred_t2.shape[0] >= 2
                and target_t2.shape[0] >= 2
            ):
                model_ot = ot_distance(
                    pred_t2,
                    target_t2,
                )

                baseline_ot = ot_distance(
                    z_pop,
                    target_t2,
                )

                score_t2.append(
                    model_ot / (baseline_ot + 1e-8)
                )
            else:
                skipped_t2 += 1

            del pred_t1
            del pred_t2

        torch.cuda.empty_cache()

    print(f"Skipped {skipped_t1} clones at t1")
    print(f"Skipped {skipped_t2} clones at t2")

    return {
        "normalized_ot_t1": safe_mean(score_t1),
        "normalized_ot_t2": safe_mean(score_t2),
    }


def add_normalized_ot_to_benchmark(
    benchmark_csv,
    models_dir,
    processed_dir,
):
    """Add normalized OT metrics to an existing benchmark CSV.

    Only rows without an existing normalized OT score are evaluated.

    Parameters
    ----------
    benchmark_csv
        Path to the benchmark CSV file.
    models_dir
        Directory containing trained model checkpoints.
    processed_dir
        Directory containing processed AnnData files.
    """
    benchmark_csv = Path(benchmark_csv)
    models_dir = Path(models_dir)
    processed_dir = Path(processed_dir)

    df = pd.read_csv(
        benchmark_csv,
    )

    # Initialize metric columns when they are not already present.
    for col in [
        "normalized_ot_t1",
        "normalized_ot_t2",
    ]:
        if col not in df.columns:
            df[col] = float("nan")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    for idx, row in df.iterrows():
        if pd.notna(
            row["normalized_ot_t2"]
        ):
            continue

        print(
            f"Normalized OT : "
            f"{row.method} "
            f"latent={row.latent_dim} "
            f"hidden={row.hidden_dim} "
            f"seed={row.seed}"
        )

        # Load the processed dataset and lineage pairs.
        adata = sc.read_h5ad(
            processed_dir
            / f"larry_{row.method}_{row.latent_dim}.h5ad"
        )

        pair_dict = build_pair_dict(
            adata,
        )

        _, loader = build_dataloaders(
            adata,
            pair_dict,
            latent_key="X_latent",
            seed=row.seed,
        )

        # Rebuild the model with the benchmark configuration.
        model = LatentSDE(
            latent_dim=int(
                row.latent_dim,
            ),
            hidden=int(
                row.hidden_dim,
            ),
        ).to(device)

        checkpoint = torch.load(
            models_dir
            / row.method
            / f"latent_{int(row.latent_dim)}"
            / f"hidden_{int(row.hidden_dim)}"
            / f"seed_{int(row.seed)}"
            / "best_model.pt",
            map_location=device,
        )

        # Support both full checkpoint dictionaries and raw state dictionaries.
        if isinstance(
            checkpoint,
            dict,
        ) and "model_state_dict" in checkpoint:
            model.load_state_dict(
                checkpoint["model_state_dict"],
            )
        else:
            model.load_state_dict(
                checkpoint,
            )

        model.eval()

        metrics = evaluate_normalized_ot(
            model=model,
            loader=loader,
        )

        df.loc[
            idx,
            "normalized_ot_t1",
        ] = metrics["normalized_ot_t1"]

        df.loc[
            idx,
            "normalized_ot_t2",
        ] = metrics["normalized_ot_t2"]

    df.to_csv(
        benchmark_csv,
        index=False,
    )

    print()
    print("Normalized OT added to benchmark.")