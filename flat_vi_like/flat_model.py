from __future__ import annotations

from typing import Optional

from anndata import AnnData
from scvi.model import SCVI

from flat_vi_like.flat_vae import FlatVAE


class FlatModel(SCVI):
    """FlatVI model built on top of the scVI model interface."""

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        gene_likelihood: str = "nb",
        latent_distribution: str = "normal",
        lambda_metric: float = 1.0,
        **model_kwargs,
    ):
        """Initialize the FlatVI model.

        Parameters
        ----------
        adata
            Annotated data matrix used to initialize the model.
        n_hidden
            Number of hidden units in the neural networks.
        n_latent
            Dimensionality of the latent representation.
        n_layers
            Number of hidden layers in the neural networks.
        dropout_rate
            Dropout rate used in the neural networks.
        dispersion
            Dispersion parameterization of the gene-expression likelihood.
        gene_likelihood
            Likelihood model used for gene expression.
        latent_distribution
            Distribution used for the latent representation.
        lambda_metric
            Weight of the metric regularization term.
        **model_kwargs
            Additional keyword arguments passed to the scVI model and module.
        """
        super().__init__(
            adata=adata,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            **model_kwargs,
        )

        self.module = FlatVAE(
            n_input=self.summary_stats.n_vars,
            n_batch=self.summary_stats.n_batch,
            n_labels=self.summary_stats.n_labels,
            n_continuous_cov=self.summary_stats.get(
                "n_extra_continuous_covs", 0
            ),
            n_cats_per_cov=self.summary_stats.get(
                "n_cats_per_cov", None
            ),
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            lambda_metric=lambda_metric,
            **model_kwargs,
        )

        self._model_summary_string = (
            f"FlatModel("
            f"n_latent={n_latent}, "
            f"n_hidden={n_hidden}, "
            f"lambda_metric={lambda_metric})"
        )

        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[list[str]] = None,
        continuous_covariate_keys: Optional[list[str]] = None,
        **kwargs,
    ):
        """Register AnnData fields used by the model.

        Parameters
        ----------
        adata
            Annotated data matrix used for model training.
        layer
            AnnData layer containing the expression data.
        batch_key
            Column in ``adata.obs`` identifying batches.
        labels_key
            Column in ``adata.obs`` containing cell labels.
        size_factor_key
            Column in ``adata.obs`` containing size factors.
        categorical_covariate_keys
            Columns in ``adata.obs`` containing categorical covariates.
        continuous_covariate_keys
            Columns in ``adata.obs`` containing continuous covariates.
        **kwargs
            Additional arguments passed to ``SCVI.setup_anndata``.
        """
        return super().setup_anndata(
            adata=adata,
            layer=layer,
            batch_key=batch_key,
            labels_key=labels_key,
            size_factor_key=size_factor_key,
            categorical_covariate_keys=categorical_covariate_keys,
            continuous_covariate_keys=continuous_covariate_keys,
            **kwargs,
        )

    @property
    def last_metrics(self):
        """Return the latest training metrics."""
        return self.module.last_metrics