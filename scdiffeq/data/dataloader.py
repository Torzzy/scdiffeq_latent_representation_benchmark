import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    DataLoader,
    Subset,
)

from scdiffeq.data.dataset import (
    LineageCollator,
    LineageDataset,
)


def build_dataloaders(
    adata,
    pair_dict,
    latent_key,
    batch_size=128,
    val_fraction=0.2,
    seed=0,
    num_workers=8,
):
    """Build training and validation dataloaders with clone-level splitting.

    Parameters
    ----------
    adata
        Annotated data matrix containing the latent representation.
    pair_dict
        Dictionary containing lineage pair information and clone labels.
    latent_key
        Key identifying the latent representation in ``adata``.
    batch_size
        Number of samples per batch.
    val_fraction
        Fraction of clones assigned to the validation set.
    seed
        Random seed used for the clone-level split.
    num_workers
        Number of worker processes used by the dataloaders.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Training and validation dataloaders.
    """
    unique_clones = np.unique(
        pair_dict["clone"]
    )

    train_clones, val_clones = train_test_split(
        unique_clones,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
    )

    train_idx = np.where(
        np.isin(
            pair_dict["clone"],
            train_clones,
        )
    )[0]

    val_idx = np.where(
        np.isin(
            pair_dict["clone"],
            val_clones,
        )
    )[0]

    dataset = LineageDataset(
        pair_dict,
    )

    train_dataset = Subset(
        dataset,
        train_idx,
    )

    val_dataset = Subset(
        dataset,
        val_idx,
    )

    collator = LineageCollator(
        adata,
        latent_key,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader