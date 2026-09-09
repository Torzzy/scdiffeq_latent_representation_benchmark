from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import scvi

from flat_vi_like.flat_model import FlatModel

from config import (
    RECONSTRUCTION_SEED,
)


# ============================================================
# Utilities
# ============================================================

def _to_numpy(x) -> np.ndarray:

    if hasattr(x, "toarray"):
        x = x.toarray()

    return np.asarray(x)


def _check_latent(
    z: torch.Tensor,
    name: str,
) -> torch.Tensor:

    if z.ndim != 2:
        raise ValueError(
            f"{name} must be 2D, "
            f"got shape={z.shape}."
        )

    if not torch.isfinite(z).all():
        raise FloatingPointError(
            f"{name} contains NaN or Inf."
        )

    return z


def _check_reconstruction(
    x: torch.Tensor,
    name: str,
) -> torch.Tensor:

    if x.ndim != 2:
        raise ValueError(
            f"{name} must be 2D, "
            f"got shape={x.shape}."
        )

    if not torch.isfinite(x).all():
        raise FloatingPointError(
            f"{name} contains NaN or Inf."
        )

    return x


# ============================================================
# SCVI
# ============================================================

def load_scvi(
    model_dir: Path,
    adata,
):

    scvi.model.SCVI.setup_anndata(
        adata,
        layer="spliced",
    )

    model = scvi.model.SCVI.load(
        model_dir,
        adata=adata,
        accelerator="cpu",
    )

    model.module.eval()

    return model


@torch.inference_mode()
def encode_scvi(
    model,
    adata,
    batch_size: int,
) -> torch.Tensor:

    latent = model.get_latent_representation(
        adata=adata,
        batch_size=batch_size,
    )

    z = torch.as_tensor(
        latent,
        dtype=torch.float32,
    )

    return _check_latent(
        z,
        "SCVI latent representation",
    )


@torch.inference_mode()
def decode_scvi(
    model,
    z: torch.Tensor,
    adata,
    batch_size: int,
) -> torch.Tensor:
    """
    Decode SCVI latent representations.

    The decoder is evaluated using a fixed log-library
    size. The final per-cell normalization is performed
    later by the common benchmark preprocessing.
    """

    module = model.module

    module.eval()

    dataloader = model._make_data_loader(
        adata=adata,
        batch_size=batch_size,
        shuffle=False,
    )

    reconstructed = []

    start = 0

    for tensors in dataloader:

        x = tensors["X"].to(
            module.device
        )

        n_batch = x.shape[0]

        z_batch = (
            z[start:start + n_batch]
            .to(
                device=module.device,
                dtype=torch.float32,
            )
        )

        batch_index = tensors.get(
            "batch",
            None,
        )

        if batch_index is None:

            batch_index = torch.zeros(
                (n_batch, 1),
                dtype=torch.long,
                device=module.device,
            )

        else:

            batch_index = (
                batch_index
                .long()
                .to(module.device)
            )

            if batch_index.ndim == 1:
                batch_index = (
                    batch_index.unsqueeze(-1)
                )

        # ----------------------------------------------------
        # Fixed library size
        # ----------------------------------------------------

        library = torch.zeros(
            (n_batch, 1),
            dtype=torch.float32,
            device=module.device,
        )

        outputs = module.generative(
            z=z_batch,
            library=library,
            batch_index=batch_index,
        )

        px_rate = outputs["px"].mu

        if not torch.isfinite(
            px_rate
        ).all():
            raise FloatingPointError(
                "SCVI decoder produced NaN or Inf."
            )

        px_rate = px_rate.clamp_min(0.0)

        reconstructed.append(
            px_rate.cpu()
        )

        start += n_batch

    reconstructed = torch.cat(
        reconstructed,
        dim=0,
    )

    return _check_reconstruction(
        reconstructed,
        "SCVI reconstruction",
    )


# ============================================================
# VELOVI
# ============================================================

def load_velovi(
    model_dir: Path,
    adata,
):

    scvi.external.VELOVI.setup_anndata(
        adata,
        spliced_layer="spliced",
        unspliced_layer="unspliced",
    )

    model = scvi.external.VELOVI.load(
        model_dir,
        adata=adata,
        accelerator="cpu",
    )

    model.module.eval()

    return model


@torch.inference_mode()
def encode_velovi(
    model,
    adata,
    batch_size: int,
) -> torch.Tensor:

    latent = model.get_latent_representation(
        adata=adata,
        batch_size=batch_size,
    )

    z = torch.as_tensor(
        latent,
        dtype=torch.float32,
        device=model.module.device,
    )

    return _check_latent(
        z,
        "VELOVI latent representation",
    )


@torch.inference_mode()
def decode_velovi(
    model,
    z: torch.Tensor,
    chunk_size: int = 512,
) -> torch.Tensor:
    """
    Decode VELOVI latent states into spliced expression.

    The stochastic Dirichlet sampling is retained because
    it is part of the VELOVI generative pathway.

    A fixed seed is used for reproducibility.
    """

    torch.manual_seed(
        RECONSTRUCTION_SEED
    )

    module = model.module

    module.eval()

    z = z.to(
        dtype=torch.float32
    )

    if z.ndim != 2:
        raise ValueError(
            "VELOVI latent representation must be 2D."
        )

    (
        gamma,
        beta,
        alpha,
        alpha_1,
        lambda_alpha,
    ) = module._get_rates()

    scale = F.softplus(
        module.scale_unconstr
    )

    from torch.distributions import Dirichlet

    chunks = []

    for start in range(
        0,
        z.shape[0],
        chunk_size,
    ):

        end = min(
            start + chunk_size,
            z.shape[0],
        )

        z_chunk = z[
            start:end
        ].to(
            device=module.device
        )

        (
            px_pi_alpha,
            px_rho,
            px_tau,
        ) = module.decoder(
            z_chunk
        )

        px_pi = Dirichlet(
            px_pi_alpha
        ).rsample()

        (
            mixture_dist_s,
            _,
            _,
        ) = module.get_px(
            px_pi,
            px_rho,
            px_tau,
            scale,
            gamma,
            beta,
            alpha,
            alpha_1,
            lambda_alpha,
        )

        spliced_fit = (
            mixture_dist_s.mean
        )

        if not torch.isfinite(
            spliced_fit
        ).all():
            raise FloatingPointError(
                "VELOVI decoder produced NaN or Inf."
            )

        spliced_fit = (
            spliced_fit.clamp_min(0.0)
        )

        chunks.append(
            spliced_fit.cpu()
        )

    reconstructed = torch.cat(
        chunks,
        dim=0,
    )

    return _check_reconstruction(
        reconstructed,
        "VELOVI reconstruction",
    )


# ============================================================
# PCA
# ============================================================

def load_pca(
    model_path: Path,
    adata,
):

    model_path = (
        Path(model_path)
        / "pca.pkl"
    )

    with open(
        model_path,
        "rb",
    ) as file:

        pca_info = pickle.load(file)

    from sklearn.decomposition import PCA

    pca = PCA()

    pca.components_ = np.asarray(
        adata.varm["PCs"],
        dtype=np.float32,
    ).T

    pca.n_components_ = (
        pca.components_.shape[0]
    )

    pca.n_features_in_ = (
        pca.components_.shape[1]
    )

    pca.mean_ = np.zeros(
        pca.n_features_in_,
        dtype=np.float32,
    )

    pca.explained_variance_ = np.asarray(
        pca_info["variance"],
        dtype=np.float32,
    )

    pca.explained_variance_ratio_ = np.asarray(
        pca_info["variance_ratio"],
        dtype=np.float32,
    )

    pca.singular_values_ = np.sqrt(
        pca.explained_variance_
        * (adata.shape[0] - 1)
    )

    return pca


def encode_pca(
    model,
    adata,
    target_sum: float = 1e4,
) -> torch.Tensor:

    counts = _to_numpy(
        adata.layers["spliced"]
    ).astype(
        np.float32,
        copy=False,
    )

    library_size = counts.sum(
        axis=1,
        keepdims=True,
    )

    if np.any(library_size <= 0):
        raise ValueError(
            "PCA input contains cells "
            "with zero library size."
        )

    x = (
        counts
        / library_size
        * target_sum
    )

    x = np.log1p(x)

    if "mean" not in adata.var:
        raise ValueError(
            "adata.var['mean'] is missing."
        )

    if "std" not in adata.var:
        raise ValueError(
            "adata.var['std'] is missing."
        )

    mean = np.asarray(
        adata.var["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        adata.var["std"],
        dtype=np.float32,
    )

    std = np.where(
        std == 0,
        1.0,
        std,
    )

    x_scaled = (
        x - mean
    ) / std

    latent = model.transform(
        x_scaled
    )

    z = torch.as_tensor(
        latent,
        dtype=torch.float32,
    )

    return _check_latent(
        z,
        "PCA latent representation",
    )


def decode_pca(
    model,
    z: torch.Tensor,
    adata,
) -> torch.Tensor:

    z_numpy = (
        z.detach()
        .cpu()
        .numpy()
    )

    x_scaled = model.inverse_transform(
        z_numpy
    )

    mean = np.asarray(
        adata.var["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        adata.var["std"],
        dtype=np.float32,
    )

    std = np.where(
        std == 0,
        1.0,
        std,
    )

    x_log = (
        x_scaled * std
        + mean
    )

    reconstructed = np.expm1(
        x_log
    )

    reconstructed = torch.as_tensor(
        reconstructed,
        dtype=torch.float32,
    )

    reconstructed = reconstructed.clamp_min(
        0.0
    )

    return _check_reconstruction(
        reconstructed,
        "PCA reconstruction",
    )


# ============================================================
# FlatVI
# ============================================================

def load_flatvi(
    model_dir: Path,
    adata,
):

    FlatModel.setup_anndata(
        adata,
        layer="spliced",
    )

    model = FlatModel.load(
        model_dir,
        adata=adata,
        accelerator="cpu",
    )

    model.module.eval()

    return model


@torch.inference_mode()
def encode_flatvi(
    model,
    adata,
) -> torch.Tensor:

    latent = model.get_latent_representation(
        adata=adata
    )

    z = torch.as_tensor(
        latent,
        dtype=torch.float32,
        device=model.module.device,
    )

    return _check_latent(
        z,
        "FlatVI latent representation",
    )


@torch.inference_mode()
def decode_flatvi(
    model,
    z,
    batch_index=None,
):
    """
    Decode FlatVI latent states.

    This uses the FlatVI decoder directly and returns
    its expression reconstruction before common benchmark
    normalization.
    """

    module = model.module

    (
        _,
        _,
        px_rate,
        _,
    ) = module.decoder(
        module.dispersion,
        z,
        None,
        batch_index,
    )

    px_rate = px_rate.clamp_min(
        0.0
    )

    return _check_reconstruction(
        px_rate,
        "FlatVI reconstruction",
    )