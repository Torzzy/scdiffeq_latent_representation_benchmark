from __future__ import annotations

import gc

import numpy as np
import torch

from scdiffeq.simulations.euler_maruyama import (
    EulerMaruyamaSimulator,
)

from .decode import (
    decode_gene_space,
)
from .ot import (
    GENE_OT_DISTANCES,
    gene_space_ot,
)


def safe_mean(values):
    """Compute the mean of a list of Python floats.

    Parameters
    ----------
    values
        List of numeric values.

    Returns
    -------
    float
        Mean value, or NaN if the list is empty.
    """
    if len(values) == 0:
        return float("nan")

    return sum(values) / len(values)


def _adata_px_scale(
    adata,
    indices,
    device,
    eps=1e-8,
    layer="counts",
):
    """Retrieve observed cells and normalize them to gene proportions.

    The observed cells are retrieved directly from the raw count matrix
    and are never encoded or decoded.

    Parameters
    ----------
    adata
        AnnData containing the raw count matrix.
    indices
        Integer indices of observed cells.
    device
        Torch device on which the data should be stored.
    eps
        Numerical stability constant used during normalization.
    layer
        AnnData layer containing raw non-negative counts. If ``None``,
        ``adata.X`` is used.

    Returns
    -------
    torch.Tensor
        Normalized gene-space representation with shape ``(N, genes)``.
        Each cell sums to approximately 1.

    Raises
    ------
    KeyError
        If the requested count layer is not present.
    ValueError
        If the count matrix contains non-finite, negative, or zero-total
        cells.
    """
    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    if len(indices) == 0:
        return torch.empty(
            (
                0,
                adata.n_vars,
            ),
            dtype=torch.float32,
            device=device,
        )

    if layer is None:
        x = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(
                f"Raw count layer '{layer}' not found in adata.layers. "
                f"Available layers: {list(adata.layers.keys())}"
            )

        x = adata.layers[layer]

    x = x[
        indices
    ]

    if hasattr(
        x,
        "toarray",
    ):
        x = x.toarray()

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    x = torch.as_tensor(
        x,
        dtype=torch.float32,
        device=device,
    )

    if not torch.isfinite(x).all():
        raise ValueError(
            "Raw count matrix contains non-finite values."
        )

    if torch.any(x < 0):
        raise ValueError(
            "Raw count matrix contains negative values. "
            "The selected layer must contain raw "
            "non-negative gene expression."
        )

    library = x.sum(
        dim=1,
        keepdim=True,
    )

    if torch.any(library <= 0):
        raise ValueError(
            "Raw count matrix contains cells with zero "
            "total counts."
        )

    x = x / (
        library + eps
    )

    return _check_gene_space(
        x,
        name="real observed px_scale",
    )


def _check_gene_space(
    x,
    *,
    name,
    atol=1e-5,
):
    """Validate a gene-space representation for OT evaluation.

    Parameters
    ----------
    x
        Gene-space representation with shape ``(N, genes)``.
    name
        Name used in validation error messages.
    atol
        Absolute and relative tolerance for the unit-mass check.

    Returns
    -------
    torch.Tensor
        Validated input tensor.

    Raises
    ------
    ValueError
        If the representation has an invalid shape, contains non-finite
        or negative values, or is not normalized to unit mass per cell.
    """
    if x.ndim != 2:
        raise ValueError(
            f"{name} must have shape (N, genes), "
            f"got {tuple(x.shape)}."
        )

    if not torch.isfinite(x).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    if torch.any(x < 0):
        raise ValueError(
            f"{name} contains negative values."
        )

    row_sums = x.sum(
        dim=1,
    )

    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=atol,
        rtol=atol,
    ):
        raise ValueError(
            f"{name} is not normalized as px_scale. "
            f"Expected every cell to sum to 1. "
            f"Observed min={row_sums.min().item():.6g}, "
            f"max={row_sums.max().item():.6g}."
        )

    return x


@torch.no_grad()
def evaluate_gene_space_ot(
    *,
    model,
    loader,
    method,
    adata,
    pair_dict,
    decoder_model=None,
    n_simulations=128,
    sinkhorn_epsilon=0.05,
):
    """Evaluate latent dynamics in normalized gene space using OT.

    For each configured ground cost, the function computes model OT,
    baseline OT, and normalized OT at the two evaluation time points.

    Parameters
    ----------
    model
        Trained latent SDE model used to simulate future latent states.
    loader
        DataLoader providing source latent states and future populations.
    method
        Gene-space decoding method.
    adata
        AnnData containing the observed gene expression data.
    pair_dict
        Dictionary mapping source cells to target populations.
    decoder_model
        Optional trained decoder model used by ``decode_gene_space``.
    n_simulations
        Number of simulated cells generated for each source cell.
    sinkhorn_epsilon
        Entropic regularization parameter used by the Sinkhorn OT
        computation.

    Returns
    -------
    dict
        OT metrics for each configured ground cost, including model,
        baseline, and normalized values at both evaluation time points.
        Backward-compatible L2 aliases are also included.
    """
    simulator = EulerMaruyamaSimulator(
        model,
    )

    device = next(
        model.parameters()
    ).device

    values = {}

    for distance in GENE_OT_DISTANCES:
        values[distance] = {
            "model_t1": [],
            "model_t2": [],
            "baseline_t1": [],
            "baseline_t2": [],
            "normalized_t1": [],
            "normalized_t2": [],
        }

    skipped_t1 = 0
    skipped_t2 = 0

    src_idx = np.asarray(
        pair_dict["src_idx"],
        dtype=np.int64,
    )

    t1_idx = pair_dict[
        "t1_idx"
    ]

    t2_idx = pair_dict[
        "t2_idx"
    ]

    if len(src_idx) != len(t1_idx):
        raise ValueError(
            "pair_dict has inconsistent lengths: "
            f"src_idx={len(src_idx)}, "
            f"t1_idx={len(t1_idx)}"
        )

    if len(src_idx) != len(t2_idx):
        raise ValueError(
            "pair_dict has inconsistent lengths: "
            f"src_idx={len(src_idx)}, "
            f"t2_idx={len(t2_idx)}"
        )

    # Map source cell indices to their positions in pair_dict.
    pair_position = {
        int(source_cell): position
        for position, source_cell
        in enumerate(src_idx)
    }

    n_processed = 0

    for batch in loader:
        if "src_indices" not in batch:
            raise KeyError(
                "The dataloader batch does not contain "
                "'src_indices'."
            )

        batch_src_indices = np.asarray(
            batch["src_indices"],
            dtype=np.int64,
        )

        z0 = batch[
            "z0"
        ].to(device)

        future_sets = batch[
            "future_sets"
        ]

        library = batch.get(
            "library",
            None,
        )

        batch_index = batch.get(
            "batch",
            None,
        )

        if library is not None:
            library = library.to(
                device,
            )

        if batch_index is not None:
            batch_index = batch_index.to(
                device,
            )

        if (
            len(batch_src_indices)
            != len(future_sets)
        ):
            raise ValueError(
                "Batch source-index count does not "
                "match future_sets count."
            )

        for i in range(
            len(future_sets)
        ):
            source_cell = int(
                batch_src_indices[i]
            )

            if source_cell not in pair_position:
                raise KeyError(
                    f"Source cell {source_cell} "
                    "not present in pair_dict."
                )

            pair_i = pair_position[
                source_cell
            ]

            target_indices_t1 = np.asarray(
                t1_idx[pair_i],
                dtype=np.int64,
            )

            target_indices_t2 = np.asarray(
                t2_idx[pair_i],
                dtype=np.int64,
            )

            z = z0[
                i:i + 1
            ]

            if library is not None:
                library_i = library[
                    i:i + 1
                ]
            else:
                library_i = None

            if batch_index is not None:
                batch_i = batch_index[
                    i:i + 1
                ]
            else:
                batch_i = None

            pred_t1 = (
                simulator
                .sample_population(
                    z,
                    2,
                    4,
                    n_simulations,
                )
                .squeeze(0)
            )

            pred_t2 = (
                simulator
                .continue_simulation(
                    pred_t1.unsqueeze(0),
                    4,
                    6,
                )
                .squeeze(0)
            )

            z_pop = z.expand(
                n_simulations,
                -1,
            )

            if library_i is not None:
                library_pop = library_i.expand(
                    n_simulations,
                    -1,
                )
            else:
                library_pop = None

            if batch_i is not None:
                batch_pop = batch_i.expand(
                    n_simulations,
                    -1,
                )
            else:
                batch_pop = None

            genes_t1 = decode_gene_space(
                method,
                pred_t1,
                model=decoder_model,
                adata=adata,
                library=library_pop,
                batch_index=batch_pop,
            )

            genes_t2 = decode_gene_space(
                method,
                pred_t2,
                model=decoder_model,
                adata=adata,
                library=library_pop,
                batch_index=batch_pop,
            )

            genes_initial = decode_gene_space(
                method,
                z_pop,
                model=decoder_model,
                adata=adata,
                library=library_pop,
                batch_index=batch_pop,
            )

            target_genes_t1 = _adata_px_scale(
                adata,
                target_indices_t1,
                device,
            )

            target_genes_t2 = _adata_px_scale(
                adata,
                target_indices_t2,
                device,
            )

            genes_t1 = _check_gene_space(
                genes_t1,
                name="decoded t1 gene space",
            )

            genes_t2 = _check_gene_space(
                genes_t2,
                name="decoded t2 gene space",
            )

            genes_initial = _check_gene_space(
                genes_initial,
                name="decoded initial gene space",
            )

            target_genes_t1 = _check_gene_space(
                target_genes_t1,
                name="real t1 gene space",
            )

            target_genes_t2 = _check_gene_space(
                target_genes_t2,
                name="real t2 gene space",
            )

            gene_dim = genes_initial.shape[1]

            if genes_t1.shape[1] != gene_dim:
                raise ValueError(
                    "genes_t1 gene dimension mismatch."
                )

            if genes_t2.shape[1] != gene_dim:
                raise ValueError(
                    "genes_t2 gene dimension mismatch."
                )

            if target_genes_t1.shape[1] != gene_dim:
                raise ValueError(
                    "target_genes_t1 gene dimension mismatch."
                )

            if target_genes_t2.shape[1] != gene_dim:
                raise ValueError(
                    "target_genes_t2 gene dimension mismatch."
                )

            valid_t1 = (
                genes_t1.shape[0] >= 2
                and target_genes_t1.shape[0] >= 2
            )

            if valid_t1:
                for distance in GENE_OT_DISTANCES:
                    model_ot = gene_space_ot(
                        genes_t1,
                        target_genes_t1,
                        distance=distance,
                        epsilon=sinkhorn_epsilon,
                    )

                    baseline_ot = gene_space_ot(
                        genes_initial,
                        target_genes_t1,
                        distance=distance,
                        epsilon=sinkhorn_epsilon,
                    )

                    normalized_ot = (
                        model_ot
                        / (
                            baseline_ot
                            + 1e-8
                        )
                    )

                    values[distance][
                        "model_t1"
                    ].append(
                        float(model_ot)
                    )

                    values[distance][
                        "baseline_t1"
                    ].append(
                        float(baseline_ot)
                    )

                    values[distance][
                        "normalized_t1"
                    ].append(
                        float(normalized_ot)
                    )
            else:
                skipped_t1 += 1

            valid_t2 = (
                genes_t2.shape[0] >= 2
                and target_genes_t2.shape[0] >= 2
            )

            if valid_t2:
                for distance in GENE_OT_DISTANCES:
                    model_ot = gene_space_ot(
                        genes_t2,
                        target_genes_t2,
                        distance=distance,
                        epsilon=sinkhorn_epsilon,
                    )

                    baseline_ot = gene_space_ot(
                        genes_initial,
                        target_genes_t2,
                        distance=distance,
                        epsilon=sinkhorn_epsilon,
                    )

                    normalized_ot = (
                        model_ot
                        / (
                            baseline_ot
                            + 1e-8
                        )
                    )

                    values[distance][
                        "model_t2"
                    ].append(
                        float(model_ot)
                    )

                    values[distance][
                        "baseline_t2"
                    ].append(
                        float(baseline_ot)
                    )

                    values[distance][
                        "normalized_t2"
                    ].append(
                        float(normalized_ot)
                    )
            else:
                skipped_t2 += 1

            del pred_t1
            del pred_t2
            del z_pop
            del genes_t1
            del genes_t2
            del genes_initial
            del target_genes_t1
            del target_genes_t2

            if library_pop is not None:
                del library_pop

            if batch_pop is not None:
                del batch_pop

            del target_indices_t1
            del target_indices_t2

            gc.collect()

            n_processed += 1

        del z0
        del future_sets
        del batch_src_indices

        if library is not None:
            del library

        if batch_index is not None:
            del batch_index

        gc.collect()

    print(
        f"Processed {n_processed} source cells"
    )

    print(
        f"Skipped {skipped_t1} cells at t1"
    )

    print(
        f"Skipped {skipped_t2} cells at t2"
    )

    metrics = {}

    for distance in GENE_OT_DISTANCES:
        distance_values = values[
            distance
        ]

        metrics[
            f"gene_ot_{distance}_t1"
        ] = safe_mean(
            distance_values["model_t1"]
        )

        metrics[
            f"gene_ot_{distance}_t2"
        ] = safe_mean(
            distance_values["model_t2"]
        )

        metrics[
            f"gene_baseline_ot_{distance}_t1"
        ] = safe_mean(
            distance_values["baseline_t1"]
        )

        metrics[
            f"gene_baseline_ot_{distance}_t2"
        ] = safe_mean(
            distance_values["baseline_t2"]
        )

        metrics[
            f"normalized_gene_ot_{distance}_t1"
        ] = safe_mean(
            distance_values["normalized_t1"]
        )

        metrics[
            f"normalized_gene_ot_{distance}_t2"
        ] = safe_mean(
            distance_values["normalized_t2"]
        )

    # Preserve the historical aliases, which refer to the L2 metric.
    metrics[
        "gene_ot_t1"
    ] = metrics[
        "gene_ot_l2_t1"
    ]

    metrics[
        "gene_ot_t2"
    ] = metrics[
        "gene_ot_l2_t2"
    ]

    metrics[
        "gene_baseline_ot_t1"
    ] = metrics[
        "gene_baseline_ot_l2_t1"
    ]

    metrics[
        "gene_baseline_ot_t2"
    ] = metrics[
        "gene_baseline_ot_l2_t2"
    ]

    metrics[
        "normalized_gene_ot_t1"
    ] = metrics[
        "normalized_gene_ot_l2_t1"
    ]

    metrics[
        "normalized_gene_ot_t2"
    ] = metrics[
        "normalized_gene_ot_l2_t2"
    ]

    return metrics