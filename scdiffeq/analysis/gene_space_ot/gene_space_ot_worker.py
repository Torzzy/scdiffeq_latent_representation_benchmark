from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import scanpy as sc
import scvi
import torch

from flat_vi_like.flat_model import FlatModel

from scdiffeq.data.dataloader import (
    build_dataloaders,
)
from scdiffeq.data.pair_dict import (
    build_pair_dict,
)
from scdiffeq.models.sde import LatentSDE

from .evaluate import (
    evaluate_gene_space_ot,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR = RESULTS_DIR / "models"

SDE_CHECKPOINT_DIR = RESULTS_DIR / "models"


def _load_sde_checkpoint(
    model,
    checkpoint_path,
    device,
):
    """Load model weights from an SDE checkpoint.

    Parameters
    ----------
    model
        Latent SDE model whose parameters will be loaded.
    checkpoint_path
        Path to the model checkpoint.
    device
        Device used to load the checkpoint.

    Returns
    -------
    LatentSDE
        Model with loaded parameters.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(
            checkpoint
        )

    del checkpoint

    return model


def _load_decoder(
    method,
    latent_dim,
    adata,
    device,
):
    """Load the trained decoder associated with a representation.

    Parameters
    ----------
    method
        Representation method. Supported values are ``pca``, ``scvi``,
        ``velovi``, and ``flatvi``.
    latent_dim
        Dimensionality of the latent representation.
    adata
        AnnData object associated with the representation.
    device
        Device on which the decoder should be loaded.

    Returns
    -------
    object or None
        Loaded decoder model, or ``None`` for PCA.

    Raises
    ------
    FileNotFoundError
        If the decoder directory does not exist.
    ValueError
        If the representation method is unknown.
    """
    method = method.lower()

    model_dir = (
        DATA_PROCESSED
        / "models"
        / f"{method}_{latent_dim}"
    )

    if not model_dir.exists():
        raise FileNotFoundError(
            "Decoder directory does not exist:\n"
            f"{model_dir}"
        )

    accelerator = (
        "gpu"
        if device == "cuda"
        else "cpu"
    )

    if method == "pca":
        return None

    if method == "scvi":
        scvi.model.SCVI.setup_anndata(
            adata,
            layer="counts",
        )

        decoder = scvi.model.SCVI.load(
            model_dir,
            adata=adata,
            accelerator=accelerator,
        )

        decoder.module.to(device)
        decoder.module.eval()

        return decoder

    if method == "velovi":
        scvi.external.VELOVI.setup_anndata(
            adata,
            spliced_layer="spliced",
            unspliced_layer="unspliced",
        )

        decoder = scvi.external.VELOVI.load(
            model_dir,
            adata=adata,
            accelerator=accelerator,
        )

        decoder.module.to(device)
        decoder.module.eval()

        return decoder

    if method == "flatvi":
        FlatModel.setup_anndata(
            adata,
            layer=None,
        )

        decoder = FlatModel.load(
            model_dir,
            adata=adata,
            accelerator=accelerator,
        )

        decoder.module.to(device)
        decoder.module.eval()

        return decoder

    raise ValueError(
        f"Unknown method: {method}"
    )


def _find_sde_checkpoint(
    method,
    latent_dim,
    hidden_dim,
    seed,
):
    """Find the trained SDE checkpoint for a benchmark configuration.

    Parameters
    ----------
    method
        Representation method.
    latent_dim
        Dimensionality of the latent representation.
    hidden_dim
        Number of hidden units in the SDE.
    seed
        Random seed associated with the model.

    Returns
    -------
    pathlib.Path
        Path to the model checkpoint.

    Raises
    ------
    FileNotFoundError
        If the expected checkpoint does not exist.
    """
    checkpoint = (
        SDE_CHECKPOINT_DIR
        / method
        / ("latent_" + str(latent_dim))
        / ("hidden_" + str(hidden_dim))
        / ("seed_" + str(seed))
        / "best_model.pt"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            "\n"
            "LatentSDE checkpoint not found.\n\n"
            f"Expected path:\n"
            f"{checkpoint}\n\n"
            f"method={method}\n"
            f"latent_dim={latent_dim}\n"
            f"hidden_dim={hidden_dim}\n"
            f"seed={seed}\n"
        )

    return checkpoint


def run_iteration(
    *,
    method,
    latent_dim,
    hidden_dim,
    seed,
    output_file,
):
    """Execute one benchmark iteration in an isolated worker process.

    Parameters
    ----------
    method
        Representation method.
    latent_dim
        Dimensionality of the latent representation.
    hidden_dim
        Number of hidden units in the latent SDE.
    seed
        Random seed associated with the model configuration.
    output_file
        Path where the benchmark metrics will be written.
    """
    method = str(
        method
    ).lower()

    output_file = Path(
        output_file
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n"
        f"[WORKER PID={os.getpid()}]\n"
        f"PROJECT_ROOT={PROJECT_ROOT}\n"
        f"method={method}\n"
        f"latent_dim={latent_dim}\n"
        f"hidden_dim={hidden_dim}\n"
        f"seed={seed}\n"
        f"device={device}",
        flush=True,
    )

    dataset_path = (
        DATA_PROCESSED
        / f"larry_{method}_{latent_dim}.h5ad"
    )

    print(
        f"[WORKER] dataset={dataset_path}",
        flush=True,
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset does not exist:\n"
            f"{dataset_path}"
        )

    adata = sc.read_h5ad(
        dataset_path
    )

    print(
        "[WORKER] adata loaded",
        flush=True,
    )

    pair_dict = build_pair_dict(
        adata
    )

    _, loader = build_dataloaders(
        adata,
        pair_dict,
        latent_key="X_latent",
        seed=seed,
    )

    print(
        "[WORKER] dataloader built",
        flush=True,
    )

    model = LatentSDE(
        latent_dim=latent_dim,
        hidden=hidden_dim,
    ).to(device)

    checkpoint = _find_sde_checkpoint(
        method=method,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        seed=seed,
    )

    print(
        f"[WORKER] checkpoint={checkpoint}",
        flush=True,
    )

    _load_sde_checkpoint(
        model,
        checkpoint,
        device,
    )

    model.eval()

    decoder_model = _load_decoder(
        method=method,
        latent_dim=latent_dim,
        adata=adata,
        device=device,
    )

    print(
        "[WORKER] starting evaluation",
        flush=True,
    )

    metrics = evaluate_gene_space_ot(
        model=model,
        loader=loader,
        method=method,
        adata=adata,
        pair_dict=pair_dict,
        decoder_model=decoder_model,
        n_simulations=128,
    )

    print(
        "[WORKER] evaluation finished",
        flush=True,
    )

    clean_metrics = {}

    for key, value in metrics.items():
        if hasattr(
            value,
            "item",
        ):
            value = value.item()

        clean_metrics[key] = float(
            value
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_file,
        "w",
    ) as f:
        json.dump(
            [clean_metrics],
            f,
        )

    print(
        f"[WORKER] result written to:\n"
        f"{output_file}",
        flush=True,
    )

    # Release large objects before the worker process terminates.
    del metrics
    del clean_metrics
    del decoder_model
    del model
    del loader
    del pair_dict
    del adata
    del checkpoint

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        "[WORKER] finished. Process will terminate.",
        flush=True,
    )


def main():
    """Parse command-line arguments and run one benchmark iteration."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        required=True,
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    run_iteration(
        method=args.method,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        output_file=Path(
            args.output
        ),
    )


if __name__ == "__main__":
    main()