from pathlib import Path
import time

import torch

from scdiffeq.trainer.run_epoch import run_epoch


def train_model(
    model,
    train_loader,
    val_loader,
    sinkhorn,
    output_dir,
    simulator,
    epochs=200,
    lr=1e-3,
    patience=20,
    min_relative_improvement=5e-3,
):
    """Train an SDE model with validation-based early stopping.

    Parameters
    ----------
    model
        SDE model to train.
    train_loader
        Dataloader providing training batches.
    val_loader
        Dataloader providing validation batches.
    sinkhorn
        Sinkhorn loss used to compare predicted and target populations.
    output_dir
        Directory where the best model checkpoint is saved.
    simulator
        Simulator used to generate predicted latent populations.
    epochs
        Maximum number of training epochs.
    lr
        Initial learning rate.
    patience
        Number of consecutive epochs without sufficient improvement
        before early stopping.
    min_relative_improvement
        Minimum relative validation-loss improvement required to update
        the best model checkpoint.

    Returns
    -------
    tuple
        Trained model and training history.
    """
    device = next(
        model.parameters()
    ).device

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            threshold=1e-3,
            threshold_mode="rel",
            cooldown=0,
            min_lr=1e-6,
        )
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    history = []

    time_ref = time.time()

    for epoch in range(
        epochs
    ):
        train_metrics = run_epoch(
            model,
            simulator,
            sinkhorn,
            train_loader,
            optimizer,
        )

        val_metrics = run_epoch(
            model,
            simulator,
            sinkhorn,
            val_loader,
        )

        scheduler.step(
            val_metrics["loss"]
        )

        history.append(
            {
                "epoch": epoch,
                **{
                    f"train_{k}": v
                    for k, v in train_metrics.items()
                },
                **{
                    f"val_{k}": v
                    for k, v in val_metrics.items()
                },
                "lr": optimizer.param_groups[0][
                    "lr"
                ],
            }
        )

        current_time = time.time()

        print(
            f"[{epoch}] "
            f"train={train_metrics['loss']:.5f} "
            f"val={val_metrics['loss']:.5f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"time={current_time - time_ref:.2f}s"
        )

        time_ref = current_time

        if best_loss == float(
            "inf"
        ):
            improvement = float(
                "inf"
            )
        else:
            improvement = (
                best_loss
                - val_metrics["loss"]
            ) / best_loss

        if improvement > min_relative_improvement:
            best_loss = val_metrics[
                "loss"
            ]

            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                output_dir / "best_model.pt",
            )

            print(
                "saved best model"
            )
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):
            print(
                f"Early stopping after {epoch + 1} epochs."
            )
            break

    # Restore the best validation checkpoint before returning the model.
    model.load_state_dict(
        torch.load(
            output_dir / "best_model.pt",
            map_location=device,
        )
    )

    return model, history