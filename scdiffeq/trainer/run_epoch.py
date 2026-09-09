import torch


def run_epoch(
    model,
    simulator,
    sinkhorn,
    loader,
    optimizer=None,
    n_simulations=32,
    verbose=False,
):
    """Run one training or evaluation epoch.

    Parameters
    ----------
    model
        SDE model providing drift and diffusion functions.
    simulator
        Simulator used to generate predicted latent populations.
    sinkhorn
        Sinkhorn loss used to compare predicted and target populations.
    loader
        Dataloader providing initial latent states and future populations.
    optimizer
        Optimizer used during training. If ``None``, the model is evaluated.
    n_simulations
        Number of simulated trajectories per initial state.
    verbose
        Whether to print diagnostic statistics.

    Returns
    -------
    dict
        Dictionary containing the mean epoch loss.
    """
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_batches = 0

    z_norm = []
    drift_norm = []
    sigma_norm = []
    cos_center = []

    pred_t1_norm = []
    pred_t2_norm = []

    future_norm = []

    nearest_future = []
    mean_future = []
    max_future = []

    latent_std = []
    cov_eigs = []

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for batch in loader:
            device = next(
                model.parameters()
            ).device

            z0 = batch[
                "z0"
            ].to(device)

            future_sets = batch[
                "future_sets"
            ]

            with torch.no_grad():
                latent_std.append(
                    z0.std(
                        dim=0
                    ).cpu()
                )

                future = future_sets[0][4].to(
                    device
                )

                future_norm.append(
                    future.norm(
                        dim=1
                    ).mean().item()
                )

                D = torch.cdist(
                    z0,
                    future,
                )

                nearest_future.append(
                    D.min(
                        dim=1
                    ).values.mean().item()
                )

                mean_future.append(
                    D.mean().item()
                )

                max_future.append(
                    D.max().item()
                )

                cov = torch.cov(
                    z0.T
                )

                eig = torch.linalg.eigvalsh(
                    cov
                )

                cov_eigs.append(
                    eig.cpu()
                )

            with torch.no_grad():
                t = torch.full(
                    (z0.shape[0],),
                    2.0,
                    device=device,
                )

                drift = model.f(
                    t,
                    z0,
                )

                sigma = model.g(
                    t,
                    z0,
                )

                z_norm.append(
                    torch.norm(
                        z0,
                        dim=1,
                    ).mean().item()
                )

                drift_norm.append(
                    torch.norm(
                        drift,
                        dim=1,
                    ).mean().item()
                )

                sigma_norm.append(
                    torch.norm(
                        sigma,
                        dim=1,
                    ).mean().item()
                )

                cos = (
                    torch.nn.functional.cosine_similarity(
                        drift,
                        -z0,
                        dim=1,
                    )
                )

                cos_center.append(
                    cos.mean().item()
                )

            pred_t1 = simulator.sample_population(
                z0,
                t0=2,
                t1=4,
                n_samples=n_simulations,
            )

            pred_t2 = simulator.continue_simulation(
                pred_t1,
                t0=4,
                t1=6,
            )

            with torch.no_grad():
                pred_t1_norm.append(
                    torch.norm(
                        pred_t1.reshape(
                            -1,
                            pred_t1.shape[-1],
                        ),
                        dim=1,
                    ).mean().item()
                )

                pred_t2_norm.append(
                    torch.norm(
                        pred_t2.reshape(
                            -1,
                            pred_t2.shape[-1],
                        ),
                        dim=1,
                    ).mean().item()
                )

            loss = 0.0

            B = len(
                future_sets
            )

            for i in range(B):
                target_t1 = future_sets[i][4].to(
                    device
                )

                target_t2 = future_sets[i][6].to(
                    device
                )

                loss += sinkhorn(
                    pred_t1[i],
                    target_t1,
                )

                loss += sinkhorn(
                    pred_t2[i],
                    target_t2,
                )

            loss /= (
                2 * B
            )

            if training:
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    if verbose:
        std = torch.stack(
            latent_std
        ).mean(
            0
        )

        eig = torch.stack(
            cov_eigs
        ).mean(
            0
        )

        print(
            "\nLATENT GEOMETRY"
        )

        print(
            "future ||z|| :",
            sum(future_norm)
            / len(future_norm),
        )

        print()

        print(
            "nearest future distance :",
            sum(nearest_future)
            / len(nearest_future),
        )

        print(
            "mean future distance    :",
            sum(mean_future)
            / len(mean_future),
        )

        print(
            "max future distance     :",
            sum(max_future)
            / len(max_future),
        )

        print()

        print(
            "dimension std"
        )

        print(
            std
        )

        print()

        print(
            "covariance eigenvalues"
        )

        print(
            eig
        )

        print(
            "condition number :",
            (
                eig.max()
                / (
                    eig.min()
                    + 1e-8
                )
            ).item(),
        )

        print(
            "\nEPOCH DIAGNOSTICS"
        )

        print(
            "\nInitial latent"
        )

        print(
            "||z||      :",
            sum(z_norm)
            / len(z_norm),
        )

        print(
            "||drift||  :",
            sum(drift_norm)
            / len(drift_norm),
        )

        print(
            "||sigma||  :",
            sum(sigma_norm)
            / len(sigma_norm),
        )

        print(
            "drift/sigma:",
            (
                sum(drift_norm)
                / len(drift_norm)
            )
            / (
                sum(sigma_norm)
                / len(sigma_norm)
                + 1e-8
            ),
        )

        print(
            "cos(drift,-z):",
            sum(cos_center)
            / len(cos_center),
        )

        print(
            "\nPredictions"
        )

        print(
            "||pred t1||:",
            sum(pred_t1_norm)
            / len(pred_t1_norm),
        )

        print(
            "||pred t2||:",
            sum(pred_t2_norm)
            / len(pred_t2_norm),
        )

    return {
        "loss": total_loss / n_batches,
    }