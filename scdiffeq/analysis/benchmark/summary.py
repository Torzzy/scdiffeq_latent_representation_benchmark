from pathlib import Path

import pandas as pd


def load_benchmark(
    benchmark_csv,
):
    """Load benchmark results from a CSV file.

    Parameters
    ----------
    benchmark_csv
        Path to the benchmark CSV file.

    Returns
    -------
    pandas.DataFrame
        Benchmark results.
    """
    return pd.read_csv(
        Path(benchmark_csv),
    )


def summarize(
    raw_df,
):
    """Compute mean and standard deviation across seeds.

    Parameters
    ----------
    raw_df
        Benchmark dataframe containing one row per seed.

    Returns
    -------
    pandas.DataFrame
        Summary dataframe grouped by method, latent dimension, and hidden
        dimension, with mean and standard deviation for numeric metrics.
    """
    group_cols = [
        "method",
        "latent_dim",
        "hidden_dim",
    ]

    numeric = raw_df.select_dtypes("number").columns.difference(
        group_cols,
    )

    summary = (
        raw_df
        .groupby(group_cols)[numeric]
        .agg(["mean", "std"])
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary = summary.reset_index()

    return summary


def load_and_summarize(
    benchmark_csv,
):
    """Load benchmark results and compute summary statistics.

    Parameters
    ----------
    benchmark_csv
        Path to the benchmark CSV file.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Raw benchmark dataframe and summarized benchmark dataframe.
    """
    raw_df = load_benchmark(
        benchmark_csv,
    )

    summary_df = summarize(
        raw_df,
    )

    return raw_df, summary_df