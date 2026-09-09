import numpy as np
import torch
from torch.utils.data import (
    Dataset,
)


class LineageDataset(Dataset):
    """Dataset containing lineage source cells and future cell indices."""

    def __init__(
        self,
        pair_dict,
    ):
        """Initialize the lineage dataset.

        Parameters
        ----------
        pair_dict
            Dictionary containing source and future cell indices.
        """
        self.src = pair_dict[
            "src_idx"
        ]

        self.t1 = pair_dict[
            "t1_idx"
        ]

        self.t2 = pair_dict[
            "t2_idx"
        ]

    def __len__(self):
        """Return the number of lineage samples."""
        return len(
            self.src
        )

    def __getitem__(
        self,
        idx,
    ):
        """Return a source cell and its future cell indices.

        Parameters
        ----------
        idx
            Sample index.

        Returns
        -------
        tuple
            Source cell index and dictionary of future cell indices
            keyed by evaluation time.
        """
        return (
            self.src[idx],
            {
                4: self.t1[idx],
                6: self.t2[idx],
            },
        )


class LineageCollator:
    """Collate lineage samples into model-ready tensors."""

    def __init__(
        self,
        adata,
        latent_key,
    ):
        """Initialize the lineage collator.

        Parameters
        ----------
        adata
            Annotated data matrix containing the latent representation.
        latent_key
            Key in ``adata.obsm`` containing the latent representation.
        """
        self.Z = np.asarray(
            adata.obsm[latent_key],
            dtype=np.float32,
        )

    def __call__(
        self,
        batch,
    ):
        """Collate a batch of lineage samples.

        Parameters
        ----------
        batch
            List of source indices and future target dictionaries.

        Returns
        -------
        dict
            Batch containing source indices, initial latent states, and
            future latent populations.
        """
        src_indices = []
        z0 = []
        future_sets = []

        for src, targets in batch:
            src_indices.append(
                src
            )

            z0.append(
                self.Z[src]
            )

            sample_targets = {}

            for time, idx in targets.items():
                sample_targets[time] = (
                    torch.from_numpy(
                        self.Z[idx]
                    )
                )

            future_sets.append(
                sample_targets
            )

        return {
            "src_indices": np.asarray(
                src_indices,
                dtype=np.int64,
            ),
            "z0": torch.from_numpy(
                np.stack(
                    z0
                )
            ),
            "future_sets": future_sets,
        }