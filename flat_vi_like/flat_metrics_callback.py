from lightning.pytorch.callbacks import Callback


class FlatMetricsCallback(Callback):

    def on_train_epoch_end(self, trainer, pl_module):

        m = pl_module.module.last_metrics

        print(
            f"[Epoch {trainer.current_epoch + 1}] "
            f"loss={m['total_loss']:.3f} | "
            f"recon={m['reconstruction_loss']:.3f} | "
            f"kl={m['kl_loss']:.3f} | "
            f"metric={m['metric_loss']:.5f}"
        )