import pandas as pd
from scipy.stats import pearsonr, spearmanr


def compute_reconstruction_prediction_correlations(
):

    """
    Compute correlations between reconstruction quality and
    future cell-state prediction performance.

    Parameters
    ----------
    reconstruction_path : str
        Path to the reconstruction CSV.

    prediction_path : str
        Path to the gene-expression prediction OT CSV.

    output_file : str
        Path where the correlation results CSV will be saved.

    Notes
    -----
    Reconstruction CSV must contain:
        method
        latent_dim
        l2_mean
        cosine_mean
        hellinger_mean

    Prediction CSV must contain:
        method
        latent_dim
        hidden_dim
        seed
        normalized_gene_ot_l2_t1
        normalized_gene_ot_l2_t2
        normalized_gene_ot_cosine_t1
        normalized_gene_ot_cosine_t2
        normalized_gene_ot_hellinger_t1
        normalized_gene_ot_hellinger_t2
    """

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    reconstruction_path = '../../results/analysis/reconstruction_evaluation/reconstruction_metrics.csv'
    prediction_path = '../../results/analysis/gene_space_ot/results.csv'

    output_file = '../../results/analysis/correlations.csv'

    reconstruction = pd.read_csv(reconstruction_path)
    prediction = pd.read_csv(prediction_path)

    print(f"Loaded reconstruction results: {len(reconstruction)} rows")
    print(f"Loaded prediction results: {len(prediction)} rows")

    # ---------------------------------------------------------
    # 2. Clean method names
    # ---------------------------------------------------------

    reconstruction["method"] = reconstruction["method"].str.upper()
    prediction["method"] = prediction["method"].str.upper()

    # Make sure latent dimension is numeric
    reconstruction["latent_dim"] = pd.to_numeric(
        reconstruction["latent_dim"]
    )

    prediction["latent_dim"] = pd.to_numeric(
        prediction["latent_dim"]
    )

    # ---------------------------------------------------------
    # 3. Average prediction performance
    #
    # One prediction representation has several hidden_dim
    # and seed configurations.
    #
    # We therefore average them so that each representation
    # (method, latent_dim) corresponds to one prediction score.
    # ---------------------------------------------------------

    prediction_metrics = [
        "normalized_gene_ot_l2_t1",
        "normalized_gene_ot_l2_t2",
        "normalized_gene_ot_cosine_t1",
        "normalized_gene_ot_cosine_t2",
        "normalized_gene_ot_hellinger_t1",
        "normalized_gene_ot_hellinger_t2",
    ]

    prediction_grouped = (
        prediction
        .groupby(["method", "latent_dim"], as_index=False)[prediction_metrics]
        .mean()
    )

    # ---------------------------------------------------------
    # 4. Merge reconstruction and prediction results
    #
    # Result:
    # 4 methods × 5 latent dimensions = 20 rows
    # ---------------------------------------------------------

    merged = pd.merge(
        reconstruction,
        prediction_grouped,
        on=["method", "latent_dim"],
        how="inner",
    )

    print(f"Merged results: {len(merged)} rows")

    expected_rows = 4 * 5

    if len(merged) != expected_rows:
        print(
            f"Warning: expected {expected_rows} representations, "
            f"but found {len(merged)}."
        )

    # ---------------------------------------------------------
    # 5. Define metrics
    # ---------------------------------------------------------

    metrics = {
        "L2": {
            "reconstruction": "l2_mean",
            "prediction": {
                "t1": "normalized_gene_ot_l2_t1",
                "t2": "normalized_gene_ot_l2_t2",
            },
        },
        "Cosine": {
            "reconstruction": "cosine_mean",
            "prediction": {
                "t1": "normalized_gene_ot_cosine_t1",
                "t2": "normalized_gene_ot_cosine_t2",
            },
        },
        "Hellinger": {
            "reconstruction": "hellinger_mean",
            "prediction": {
                "t1": "normalized_gene_ot_hellinger_t1",
                "t2": "normalized_gene_ot_hellinger_t2",
            },
        },
    }

    # ---------------------------------------------------------
    # 6. Compute correlations
    # ---------------------------------------------------------

    results = []

    for metric_name, metric_info in metrics.items():

        reconstruction_col = metric_info["reconstruction"]

        for time_point, prediction_col in metric_info["prediction"].items():

            x = merged[reconstruction_col]
            y = merged[prediction_col]

            # Remove missing values
            valid = x.notna() & y.notna()

            x_valid = x[valid]
            y_valid = y[valid]

            # Pearson
            pearson_r, pearson_p = pearsonr(
                x_valid,
                y_valid,
            )

            # Spearman
            spearman_rho, spearman_p = spearmanr(
                x_valid,
                y_valid,
            )

            results.append({
                "analysis": "global",
                "method": "ALL",
                "metric": metric_name,
                "time": time_point,
                "n": len(x_valid),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
            })

            # -------------------------------------------------
            # 7. Within-method correlations
            # -------------------------------------------------

            for method in sorted(merged["method"].unique()):

                subset = merged[
                    merged["method"] == method
                ]

                x_method = subset[reconstruction_col]
                y_method = subset[prediction_col]

                valid_method = (
                    x_method.notna()
                    & y_method.notna()
                )

                x_method = x_method[valid_method]
                y_method = y_method[valid_method]

                # With 5 latent dimensions, correlation is
                # possible but should be interpreted cautiously.
                if len(x_method) >= 3:

                    pearson_method_r, pearson_method_p = pearsonr(
                        x_method,
                        y_method,
                    )

                    spearman_method_rho, spearman_method_p = spearmanr(
                        x_method,
                        y_method,
                    )

                else:
                    pearson_method_r = float("nan")
                    pearson_method_p = float("nan")
                    spearman_method_rho = float("nan")
                    spearman_method_p = float("nan")

                results.append({
                    "analysis": "within_method",
                    "method": method,
                    "metric": metric_name,
                    "time": time_point,
                    "n": len(x_method),
                    "pearson_r": pearson_method_r,
                    "pearson_p": pearson_method_p,
                    "spearman_rho": spearman_method_rho,
                    "spearman_p": spearman_method_p,
                })

    # ---------------------------------------------------------
    # 8. Save correlation results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output_file,
        index=False,
    )

    print(f"\nSaved correlation results to:")
    print(output_file)

    # ---------------------------------------------------------
    # 9. Print global results
    # ---------------------------------------------------------

    print("\nGlobal correlations:")
    print(
        results_df[
            results_df["analysis"] == "global"
        ].to_string(index=False)
    )

    return merged, results_df

if __name__ == "__main__":
    compute_reconstruction_prediction_correlations()