import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch.utils.data import (
    DataLoader,
    Dataset,
)


class MeanPredictionDataset(Dataset):
    """Dataset for latent mean prediction."""

    def __init__(
        self,
        X,
        Y,
    ):
        """Initialize the dataset.

        Parameters
        ----------
        X
            Input latent representations.
        Y
            Target latent mean representations.
        """
        self.X = torch.tensor(
            X,
            dtype=torch.float32,
        )

        self.Y = torch.tensor(
            Y,
            dtype=torch.float32,
        )

    def __len__(self):
        """Return the number of samples."""
        return len(
            self.X
        )

    def __getitem__(
        self,
        idx,
    ):
        """Return one input-target pair.

        Parameters
        ----------
        idx
            Sample index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Input latent representation and target latent mean.
        """
        return (
            self.X[idx],
            self.Y[idx],
        )


class MeanMLP(nn.Module):
    """MLP predicting the future mean latent representation."""

    def __init__(
        self,
        latent_dim,
        hidden,
    ):
        """Initialize the mean prediction network.

        Parameters
        ----------
        latent_dim
            Dimensionality of the latent representation.
        hidden
            Number of hidden units in each hidden layer.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                latent_dim,
                hidden,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden,
                hidden,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden,
                latent_dim,
            ),
        )

    def forward(
        self,
        x,
    ):
        """Predict the future mean latent representation.

        Parameters
        ----------
        x
            Input latent representation.

        Returns
        -------
        torch.Tensor
            Predicted future mean latent representation.
        """
        return self.net(
            x
        )


def build_mean_dataset(
    adata,
    pair_dict,
    latent_key="X_latent",
):
    """Build a dataset of initial states and future latent means.

    Parameters
    ----------
    adata
        Annotated data matrix containing the latent representation.
    pair_dict
        Dictionary containing source and target cell indices.
    latent_key
        Key in ``adata.obsm`` containing the latent representation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Input latent representations and corresponding future means.
    """
    latent = np.asarray(
        adata.obsm[latent_key]
    )

    X = []
    Y = []

    for src, t2 in zip(
        pair_dict["src_idx"],
        pair_dict["t2_idx"],
    ):
        X.append(
            latent[src]
        )

        Y.append(
            latent[t2].mean(
                axis=0
            )
        )

    return (
        np.asarray(
            X,
            dtype=np.float32,
        ),
        np.asarray(
            Y,
            dtype=np.float32,
        ),
    )


def split_dataset(
    X,
    Y,
    seed,
):
    """Split a dataset into training and validation subsets.

    Parameters
    ----------
    X
        Input samples.
    Y
        Target samples.
    seed
        Random seed used for the split.

    Returns
    -------
    tuple[np.ndarray, ...]
        Training inputs, training targets, validation inputs, and
        validation targets.
    """
    rng = np.random.RandomState(
        seed
    )

    perm = rng.permutation(
        len(X)
    )

    split = int(
        0.8 * len(X)
    )

    train = perm[
        :split
    ]

    val = perm[
        split:
    ]

    return (
        X[train],
        Y[train],
        X[val],
        Y[val],
    )


def train_mean_predictor(
    X_train,
    Y_train,
    latent_dim,
    hidden_dim,
    device,
    epochs=200,
    lr=1e-3,
    batch_size=512,
):
    """Train an MLP to predict future latent means.

    Parameters
    ----------
    X_train
        Training input latent representations.
    Y_train
        Training target latent means.
    latent_dim
        Dimensionality of the latent representation.
    hidden_dim
        Number of hidden units in the MLP.
    device
        Device used for training.
    epochs
        Number of training epochs.
    lr
        Learning rate for the Adam optimizer.
    batch_size
        Number of samples per training batch.

    Returns
    -------
    MeanMLP
        Trained mean prediction model.
    """
    train_ds = MeanPredictionDataset(
        X_train,
        Y_train,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = MeanMLP(
        latent_dim,
        hidden_dim,
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    for epoch in range(
        epochs
    ):
        model.train()

        for x, y in train_loader:
            x = x.to(
                device
            )

            y = y.to(
                device
            )

            pred = model(
                x
            )

            loss = F.mse_loss(
                pred,
                y,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        print(
            f"Epoch {epoch}"
        )

    return model


@torch.no_grad()
def evaluate_mean_predictor(
    model,
    X_val,
    Y_val,
    device,
):
    """Evaluate a latent mean prediction model.

    Parameters
    ----------
    model
        Trained mean prediction model.
    X_val
        Validation input latent representations.
    Y_val
        Validation target latent means.
    device
        Device used for inference.

    Returns
    -------
    dict
        Prediction metrics including MSE, MAE, R², baseline MSE, and
        relative MSE.
    """
    model.eval()

    pred = model(
        torch.tensor(
            X_val,
            dtype=torch.float32,
            device=device,
        )
    ).cpu().numpy()

    mse = mean_squared_error(
        Y_val,
        pred,
    )

    mae = mean_absolute_error(
        Y_val,
        pred,
    )

    r2 = r2_score(
        Y_val,
        pred,
    )

    baseline = np.repeat(
        Y_val.mean(
            axis=0,
            keepdims=True,
        ),
        len(Y_val),
        axis=0,
    )

    baseline_mse = mean_squared_error(
        Y_val,
        baseline,
    )

    return {
        "mse": float(
            mse
        ),
        "mae": float(
            mae
        ),
        "r2": float(
            r2
        ),
        "baseline_mse": float(
            baseline_mse
        ),
        "relative_mse": float(
            mse / baseline_mse
        ),
    }