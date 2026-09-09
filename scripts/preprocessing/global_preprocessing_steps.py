import scanpy as sc


def preprocess_adata(
    adata,
    n_hvg=2000,
):
    """Preprocess an AnnData object and select highly variable genes.

    Parameters
    ----------
    adata
        Annotated data matrix to preprocess.
    n_hvg
        Number of highly variable genes to retain.

    Returns
    -------
    AnnData
        Preprocessed AnnData object containing only highly variable genes.
    """
    adata.layers["counts"] = adata.X.copy()

    sc.pp.filter_genes(
        adata,
        min_cells=3,
    )

    sc.pp.highly_variable_genes(
        adata,
        layer="counts",
        flavor="seurat_v3",
        n_top_genes=n_hvg,
    )

    adata = adata[
        :,
        adata.var.highly_variable,
    ].copy()

    return adata