from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _as_tensor(
    x,
    device,
    dtype=torch.float32,
):
    """Convert NumPy, SciPy, or PyTorch input to a tensor.

    Parameters
    ----------
    x
        Input array or tensor.
    device
        Device on which the tensor should be created.
    dtype
        Data type of the resulting tensor.

    Returns
    -------
    torch.Tensor
        Converted tensor.
    """
    if torch.is_tensor(x):
        return x.to(
            device=device,
            dtype=dtype,
        )

    if hasattr(x, "toarray"):
        x = x.toarray()

    return torch.as_tensor(
        x,
        device=device,
        dtype=dtype,
    )


def _check_gene_space(
    x,
    name="gene_space",
):
    """Validate a decoded gene-space representation.

    Parameters
    ----------
    x
        Gene-space tensor to validate.
    name
        Name used in error messages.

    Returns
    -------
    torch.Tensor
        Validated input tensor.

    Raises
    ------
    FloatingPointError
        If the tensor contains NaN, Inf, or negative values.
    """
    if not torch.isfinite(x).all():
        raise FloatingPointError(
            f"{name} contains NaN/Inf."
        )

    if (x < 0).any():
        raise FloatingPointError(
            f"{name} contains negative values."
        )

    return x


def load_pca_model(
    model_path,
):
    """Load PCA metadata saved during PCA training.

    Parameters
    ----------
    model_path
        Path to the serialized PCA metadata.

    Returns
    -------
    object
        Deserialized PCA metadata.
    """
    model_path = Path(model_path)

    with open(
        model_path,
        "rb",
    ) as f:
        pca = pickle.load(f)

    return pca


def decode_pca(
    z,
    adata,
    *,
    pca_model_path=None,
):
    """Decode PCA latent coordinates into normalized gene space.

    The inverse transformation applies the PCA inverse projection,
    inverse scaling, inverse ``log1p``, and per-cell normalization.

    Parameters
    ----------
    z
        Latent representation with shape ``(N, latent_dim)``.
    adata
        Processed AnnData containing the PCA components and scaling
        statistics in ``varm["PCs"]``, ``var["mean"]``, and ``var["std"]``.
    pca_model_path
        Optional path to the serialized PCA metadata.

    Returns
    -------
    torch.Tensor
        Normalized gene-space representation with shape ``(N, genes)``.
        Each cell sums to approximately 1.

    Raises
    ------
    ValueError
        If the PCA metadata or AnnData fields are missing or incompatible.
    FloatingPointError
        If the decoded representation contains cells with zero total
        expression or invalid values.
    """
    device = z.device

    if pca_model_path is not None:
        pca_model = load_pca_model(
            pca_model_path,
        )

        if not isinstance(
            pca_model,
            dict,
        ):
            raise ValueError(
                "Unexpected PCA pickle format."
            )

    if "PCs" not in adata.varm:
        raise ValueError(
            "adata.varm['PCs'] is missing. "
            "PCA components are required."
        )

    components = adata.varm["PCs"]

    if hasattr(
        components,
        "toarray",
    ):
        components = components.toarray()

    components = np.asarray(
        components,
        dtype=np.float32,
    )

    components = torch.as_tensor(
        components,
        device=device,
        dtype=torch.float32,
    )

    if z.ndim != 2:
        raise ValueError(
            f"Expected z to have shape (N, latent_dim), "
            f"got {tuple(z.shape)}."
        )

    if z.shape[1] != components.shape[1]:
        raise ValueError(
            "PCA latent dimension mismatch: "
            f"z has {z.shape[1]}, "
            f"components expect {components.shape[1]}."
        )

    x_scaled = (
        z
        @ components.T
    )

    if "mean" not in adata.var:
        raise ValueError(
            "adata.var['mean'] is missing. "
            "Cannot invert sc.pp.scale() exactly."
        )

    if "std" not in adata.var:
        raise ValueError(
            "adata.var['std'] is missing. "
            "Cannot invert sc.pp.scale() exactly."
        )

    mean = _as_tensor(
        np.asarray(
            adata.var["mean"],
            dtype=np.float32,
        ),
        device=device,
    )

    std = _as_tensor(
        np.asarray(
            adata.var["std"],
            dtype=np.float32,
        ),
        device=device,
    )

    # Avoid division by zero for zero-variance genes.
    std = torch.where(
        std > 0,
        std,
        torch.ones_like(std),
    )

    x_log = (
        x_scaled * std
        + mean
    )

    x = torch.expm1(
        x_log,
    )

    x = torch.clamp(
        x,
        min=0.0,
    )

    library = x.sum(
        dim=1,
        keepdim=True,
    )

    if (library <= 0).any():
        raise FloatingPointError(
            "Decoded PCA contains cells with zero "
            "total gene expression."
        )

    x = x / library

    return _check_gene_space(
        x,
        name="decoded PCA normalized gene space",
    )


def decode_scvi(
    model,
    z,
):
    """Decode SCVI latent states into normalized gene space.

    The model is assumed to have been trained without an explicit
    batch key, so all cells are decoded using batch index 0.

    Parameters
    ----------
    model
        Trained SCVI model.
    z
        Latent representation with shape ``(N, latent_dim)``.

    Returns
    -------
    torch.Tensor
        Normalized gene-space representation.
    """
    module = model.module

    batch_index = torch.zeros(
        z.shape[0],
        1,
        device=z.device,
        dtype=torch.long,
    )

    library = torch.zeros(
        z.shape[0],
        1,
        device=z.device,
        dtype=z.dtype,
    )

    (
        px_scale,
        _,
        _,
        _,
    ) = module.decoder(
        module.dispersion,
        z,
        library,
        batch_index,
    )

    return _check_gene_space(
        px_scale,
        name="decoded SCVI normalized gene space",
    )


@torch.no_grad()
def decode_velovi(
    model,
    z,
    latent_dim=None,
):
    """Decode VELOVI latent states into normalized spliced expression.

    The function follows the VELOVI decoder and generative path, then
    uses the mean of the fitted spliced mixture distribution. Negative
    fitted values are reported, clamped to zero, and the resulting
    expression profiles are normalized per cell.

    Parameters
    ----------
    model
        Loaded VELOVI model.
    z
        Latent representation with shape ``(N, latent_dim)``.
    latent_dim
        Optional latent dimension restriction.

    Returns
    -------
    torch.Tensor
        Clamped and normalized fitted spliced expression with shape
        ``(N, genes)``.
    """
    module = model.module

    module.eval()

    z = z.to(
        device=module.device,
        dtype=torch.float32,
    )

    (
        px_pi_alpha,
        px_rho,
        px_tau,
    ) = module.decoder(
        z,
        latent_dim=latent_dim,
    )

    from torch.distributions import Dirichlet

    if px_pi_alpha.device.type == "mps":
        px_pi = (
            Dirichlet(
                px_pi_alpha.cpu()
            )
            .rsample()
            .to(px_pi_alpha.device)
        )
    else:
        px_pi = (
            Dirichlet(
                px_pi_alpha
            )
            .rsample()
        )

    (
        gamma,
        beta,
        alpha,
        alpha_1,
        lambda_alpha,
    ) = module._get_rates()

    scale = F.softplus(
        module.scale_unconstr,
    )

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

    spliced_fit = mixture_dist_s.mean

    if not torch.isfinite(
        spliced_fit
    ).all():
        raise FloatingPointError(
            "VELOVI decoder produced NaN or Inf."
        )

    negative_mask = (
        spliced_fit < 0
    )

    n_negative = (
        negative_mask.sum().item()
    )

    if n_negative > 0:
        n_total = spliced_fit.numel()

        negative_fraction = (
            100.0
            * n_negative
            / n_total
        )

        min_value = (
            spliced_fit.min().item()
        )

        print(
            "\n"
            "[VELOVI] Negative spliced "
            "decoder values detected"
        )

        print(
            f"[VELOVI] negative values : "
            f"{n_negative} / {n_total} "
            f"({negative_fraction:.6f}%)"
        )

        print(
            f"[VELOVI] minimum value   : "
            f"{min_value:.10f}"
        )

    spliced_fit = (
        spliced_fit.clamp_min(0.0)
    )

    normalization = (
        spliced_fit.abs()
        .sum(
            dim=-1,
            keepdim=True,
        )
    )

    px_scale = (
        spliced_fit
        / (
            normalization
            + 1e-8
        )
    )

    if not torch.isfinite(
        px_scale
    ).all():
        raise FloatingPointError(
            "VELOVI normalized decoder output "
            "contains NaN or Inf."
        )

    if (
        px_scale < 0
    ).any():
        raise FloatingPointError(
            "VELOVI normalized decoder output "
            "contains negative values after clamping."
        )

    return px_scale


def decode_flatvi(
    model,
    z,
    batch_index=None,
):
    """Decode FlatVI latent states into normalized gene space.

    Parameters
    ----------
    model
        Trained FlatVI model.
    z
        Latent representation with shape ``(N, latent_dim)``.
    batch_index
        Optional batch covariate tensor.

    Returns
    -------
    torch.Tensor
        Normalized gene-space representation with shape ``(N, genes)``.
    """
    module = model.module

    px_scale, _, _, _ = module.decoder(
        module.dispersion,
        z,
        None,
        batch_index,
    )

    return _check_gene_space(
        px_scale,
        name="decoded FlatVI px_scale",
    )


def decode_gene_space(
    method,
    z,
    *,
    model=None,
    adata=None,
    library=None,
    batch_index=None,
    pca_model_path=None,
):
    """Decode latent states into a common gene-space representation.

    Parameters
    ----------
    method
        Decoding method. Supported values are ``"pca"``, ``"scvi"``,
        ``"velovi"``, and ``"flatvi"``.
    z
        Latent representation with shape ``(N, latent_dim)``.
    model
        Trained SCVI, VELOVI, or FlatVI model.
    adata
        AnnData associated with the representation. Required for PCA.
    library
        Library-size tensor for SCVI-like models.
    batch_index
        Batch covariate tensor for FlatVI.
    pca_model_path
        Path to the serialized PCA metadata.

    Returns
    -------
    torch.Tensor
        Gene-space representation with shape ``(N, genes)``.

    Raises
    ------
    ValueError
        If required inputs are missing or the decoding method is unknown.
    """
    method = method.lower()

    if method == "pca":
        if adata is None:
            raise ValueError(
                "adata is required for PCA decoding."
            )

        return decode_pca(
            z=z,
            adata=adata,
            pca_model_path=pca_model_path,
        )

    if model is None:
        raise ValueError(
            f"A trained model is required "
            f"for method={method}."
        )

    if method == "scvi":
        return decode_scvi(
            model=model,
            z=z,
        )

    if method == "velovi":
        return decode_velovi(
            model=model,
            z=z,
        )

    if method == "flatvi":
        return decode_flatvi(
            model=model,
            z=z,
            batch_index=batch_index,
        )

    raise ValueError(
        f"Unknown decoding method: {method}"
    )