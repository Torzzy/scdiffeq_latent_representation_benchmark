import pandas as pd

from scdiffeq.analysis.utils import (
    available_metrics,
    ensure_dir,
    metric_mean,
    metric_std,
)


def make_tables(
    raw_df,
    summary_df,
    output_dir,
):
    """Generate benchmark summary and ranking tables.

    Parameters
    ----------
    raw_df
        Raw benchmark dataframe containing one row per seed.
    summary_df
        Benchmark dataframe summarized across seeds.
    output_dir
        Directory where generated tables will be saved.
    """
    output_dir = ensure_dir(output_dir)

    summary_df.to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    metrics = available_metrics(summary_df)

    rows = []

    for (
        latent_dim,
        hidden_dim,
    ), df in summary_df.groupby(
        ["latent_dim", "hidden_dim"]
    ):
        row = {
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
        }

        if "ratio_mean" in metrics:
            best = df.loc[
                df[
                    metric_mean("ratio_mean")
                ].idxmax()
            ]

            row["best_ratio_method"] = best["method"]
            row["best_ratio"] = best[
                metric_mean("ratio_mean")
            ]

        if "normalized_ot_t2" in metrics:
            best = df.loc[
                df[
                    metric_mean("normalized_ot_t2")
                ].idxmin()
            ]

            row["best_normalized_ot_method"] = best[
                "method"
            ]

            row["best_normalized_ot"] = best[
                metric_mean("normalized_ot_t2")
            ]

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        output_dir / "best_models.csv",
        index=False,
    )

    if "ratio_mean" in metrics:
        rows = []

        for (
            latent_dim,
            hidden_dim,
        ), df in summary_df.groupby(
            ["latent_dim", "hidden_dim"]
        ):
            ordered = df.sort_values(
                metric_mean("ratio_mean"),
                ascending=False,
            )

            for rank, (_, r) in enumerate(
                ordered.iterrows(),
                start=1,
            ):
                rows.append(
                    {
                        "latent_dim": latent_dim,
                        "hidden_dim": hidden_dim,
                        "rank": rank,
                        "method": r["method"],
                        "ratio_mean": r[
                            metric_mean("ratio_mean")
                        ],
                        "ratio_std": r[
                            metric_std("ratio_mean")
                        ],
                    }
                )

        pd.DataFrame(rows).to_csv(
            output_dir / "ratio_ranking.csv",
            index=False,
        )

    if "normalized_ot_t2" in metrics:
        rows = []

        for (
            latent_dim,
            hidden_dim,
        ), df in summary_df.groupby(
            ["latent_dim", "hidden_dim"]
        ):
            ordered = df.sort_values(
                metric_mean("normalized_ot_t2"),
                ascending=True,
            )

            for rank, (_, r) in enumerate(
                ordered.iterrows(),
                start=1,
            ):
                rows.append(
                    {
                        "latent_dim": latent_dim,
                        "hidden_dim": hidden_dim,
                        "rank": rank,
                        "method": r["method"],
                        "normalized_ot_t2": r[
                            metric_mean("normalized_ot_t2")
                        ],
                        "normalized_ot_t2_std": r[
                            metric_std("normalized_ot_t2")
                        ],
                    }
                )

        pd.DataFrame(rows).to_csv(
            output_dir / "normalized_ot_ranking.csv",
            index=False,
        )

    (
        raw_df
        .groupby(
            [
                "method",
                "latent_dim",
            ],
            as_index=False,
        )
        .mean(
            numeric_only=True,
        )
        .to_csv(
            output_dir / "averaged_over_hidden.csv",
            index=False,
        )
    )