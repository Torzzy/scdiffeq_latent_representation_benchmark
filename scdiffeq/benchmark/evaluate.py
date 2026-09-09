from collections import defaultdict

import torch

from scdiffeq.benchmark.metrics.fate_entropy import (
    covariance_trace,
)
from scdiffeq.benchmark.metrics.latent_statistics import (
    latent_statistics,
)
from scdiffeq.benchmark.metrics.ot_distance import (
    ot_distance,
)
from scdiffeq.benchmark.metrics.reconstruction import (
    mean_future_distance,
)
from scdiffeq.simulations.euler_maruyama import (
    EulerMaruyamaSimulator,
)


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    n_simulations=128,
):
    """Evaluate an SDE model on simulated future populations.

    Parameters
    ----------
    model
        Trained SDE model used for population simulation.
    loader
        DataLoader providing initial latent states and future populations.
    n_simulations
        Number of simulated cells generated for each clone.

    Returns
    -------
    dict
        Mean evaluation metrics aggregated across batches and clones.
    """
    simulator = EulerMaruyamaSimulator(
        model,
    )

    metrics = defaultdict(
        list,
    )

    device = next(
        model.parameters()
    ).device

    for batch in loader:
        z0 = batch["z0"].to(
            device,
        )

        future_sets = batch[
            "future_sets"
        ]

        latent = latent_statistics(
            z0,
        )

        for key, value in latent.items():
            metrics[key].append(
                value
            )

        B = len(
            future_sets
        )

        for i in range(B):
            z = z0[
                i : i + 1
            ]

            target_t1 = future_sets[i][4].to(
                device,
            )

            target_t2 = future_sets[i][6].to(
                device,
            )

            pred_t1, stats_t1 = (
                simulator.sample_population_trajectory(
                    z,
                    t0=2,
                    t1=4,
                    n_samples=n_simulations,
                )
            )

            pred_t2, stats_t2 = (
                simulator.continue_trajectory(
                    pred_t1,
                    t0=4,
                    t1=6,
                )
            )

            metrics[
                "trajectory_length"
            ].append(
                stats_t1["trajectory_length"]
                + stats_t2["trajectory_length"]
            )

            metrics[
                "ratio_mean"
            ].append(
                (
                    stats_t1["ratio_mean"]
                    + stats_t2["ratio_mean"]
                )
                / 2
            )

            metrics[
                "ratio_std"
            ].append(
                (
                    stats_t1["ratio_std"]
                    + stats_t2["ratio_std"]
                )
                / 2
            )

            metrics[
                "ratio_max"
            ].append(
                max(
                    stats_t1["ratio_max"],
                    stats_t2["ratio_max"],
                )
            )

            metrics[
                "drift_mean"
            ].append(
                (
                    stats_t1["drift_mean"]
                    + stats_t2["drift_mean"]
                )
                / 2
            )

            metrics[
                "drift_max"
            ].append(
                max(
                    stats_t1["drift_max"],
                    stats_t2["drift_max"],
                )
            )

            metrics[
                "diffusion_mean"
            ].append(
                (
                    stats_t1["diffusion_mean"]
                    + stats_t2["diffusion_mean"]
                )
                / 2
            )

            metrics[
                "diffusion_max"
            ].append(
                max(
                    stats_t1["diffusion_max"],
                    stats_t2["diffusion_max"],
                )
            )

            pred_t1 = pred_t1.squeeze(
                0,
            )

            pred_t2 = pred_t2.squeeze(
                0,
            )

            metrics[
                "ot_t1"
            ].append(
                ot_distance(
                    pred_t1,
                    target_t1,
                )
            )

            metrics[
                "ot_t2"
            ].append(
                ot_distance(
                    pred_t2,
                    target_t2,
                )
            )

            metrics[
                "mean_distance_t1"
            ].append(
                mean_future_distance(
                    pred_t1,
                    target_t1,
                )
            )

            metrics[
                "mean_distance_t2"
            ].append(
                mean_future_distance(
                    pred_t2,
                    target_t2,
                )
            )

            metrics[
                "cov_trace_t1"
            ].append(
                covariance_trace(
                    pred_t1,
                )
            )

            metrics[
                "cov_trace_t2"
            ].append(
                covariance_trace(
                    pred_t2,
                )
            )

            del pred_t1
            del pred_t2

        torch.cuda.empty_cache()

    return {
        key: float(
            torch.tensor(
                value
            ).mean()
        )
        for key, value in metrics.items()
    }