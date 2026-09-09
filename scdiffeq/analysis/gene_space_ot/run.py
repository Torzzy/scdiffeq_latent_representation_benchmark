from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from .plotting import make_plots


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

BENCHMARK_CSV = (
    PROJECT_ROOT
    / "results"
    / "benchmark.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "gene_space_ot"
)

WORKER_MODULE = (
    "scdiffeq.analysis.gene_space_ot.gene_space_ot_worker"
)


GENE_OT_DISTANCES = (
    "l2",
    "cosine",
    "hellinger",
)


def _gene_ot_metric_columns():
    """Return the metrics produced by one worker.

    Returns
    -------
    list of str
        Metric names for model OT, baseline OT, and normalized OT at
        both evaluation time points for every gene-space distance.
    """
    columns = []

    for distance in GENE_OT_DISTANCES:
        columns.extend(
            [
                f"gene_ot_{distance}_t1",
                f"gene_ot_{distance}_t2",
                f"gene_baseline_ot_{distance}_t1",
                f"gene_baseline_ot_{distance}_t2",
                f"normalized_gene_ot_{distance}_t1",
                f"normalized_gene_ot_{distance}_t2",
            ]
        )

    return columns


GENE_OT_METRIC_COLUMNS = (
    _gene_ot_metric_columns()
)


def print_ram(
    label="",
):
    """Print the current resident memory usage of the parent process.

    Parameters
    ----------
    label
        Optional label included in the diagnostic message.
    """
    import psutil

    process = psutil.Process(
        os.getpid()
    )

    rss = (
        process
        .memory_info()
        .rss
    )

    print(
        f"[PARENT RAM] {label}: "
        f"{rss / 1024**3:.2f} GB",
        flush=True,
    )


def _run_isolated_iteration(
    *,
    method,
    latent_dim,
    hidden_dim,
    seed,
    output_file,
):
    """Run one benchmark configuration in a fresh OS process.

    Parameters
    ----------
    method
        Representation method.
    latent_dim
        Dimensionality of the latent representation.
    hidden_dim
        Number of hidden units in the latent SDE.
    seed
        Random seed associated with the model configuration.
    output_file
        Path where the worker writes its JSON result.

    Returns
    -------
    dict
        Metrics returned by the worker.

    Raises
    ------
    RuntimeError
        If the worker fails, does not produce an output file, or returns
        an invalid or incomplete result.
    """
    command = [
        sys.executable,
        "-m",
        WORKER_MODULE,
        "--method",
        str(method),
        "--latent-dim",
        str(latent_dim),
        "--hidden-dim",
        str(hidden_dim),
        "--seed",
        str(seed),
        "--output",
        str(output_file),
    ]

    print()
    print("=" * 80)
    print("STARTING ISOLATED PROCESS")
    print("=" * 80)

    print(
        f"method      = {method}"
    )

    print(
        f"latent_dim  = {latent_dim}"
    )

    print(
        f"hidden_dim  = {hidden_dim}"
    )

    print(
        f"seed        = {seed}"
    )

    env = os.environ.copy()

    env[
        "PYTHONUNBUFFERED"
    ] = "1"

    env[
        "OMP_NUM_THREADS"
    ] = "1"

    env[
        "MKL_NUM_THREADS"
    ] = "1"

    env[
        "OPENBLAS_NUM_THREADS"
    ] = "1"

    env[
        "NUMEXPR_NUM_THREADS"
    ] = "1"

    completed = subprocess.run(
        command,
        env=env,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "\n"
            "Isolated gene-space OT worker failed.\n\n"
            f"method={method}\n"
            f"latent_dim={latent_dim}\n"
            f"hidden_dim={hidden_dim}\n"
            f"seed={seed}\n"
            f"returncode={completed.returncode}\n"
        )

    if not output_file.exists():
        raise RuntimeError(
            "Worker exited successfully but did not "
            "produce its result file:\n"
            f"{output_file}"
        )

    with open(
        output_file,
        "r",
    ) as f:
        data = json.load(
            f
        )

    if not isinstance(
        data,
        list,
    ):
        raise RuntimeError(
            "Worker JSON result must be a list."
        )

    if len(data) != 1:
        raise RuntimeError(
            "Worker JSON result must contain "
            "exactly one row."
        )

    result = data[0]

    if not isinstance(
        result,
        dict,
    ):
        raise RuntimeError(
            "Worker JSON row must be a dictionary."
        )

    missing = [
        metric
        for metric in GENE_OT_METRIC_COLUMNS
        if metric not in result
    ]

    if missing:
        raise RuntimeError(
            "Worker did not return all gene-space "
            "OT metrics:\n"
            f"{missing}"
        )

    return result


def _configuration_mask(
    df,
    *,
    method,
    latent_dim,
    hidden_dim,
    seed,
):
    """Build a mask selecting one exact benchmark configuration.

    Parameters
    ----------
    df
        Benchmark dataframe.
    method
        Representation method.
    latent_dim
        Dimensionality of the latent representation.
    hidden_dim
        Number of hidden units in the latent SDE.
    seed
        Random seed associated with the model configuration.

    Returns
    -------
    pandas.Series
        Boolean mask identifying matching rows. If required columns
        are missing, the returned mask contains only ``False`` values.
    """
    if len(df) == 0:
        return pd.Series(
            False,
            index=df.index,
            dtype=bool,
        )

    required = [
        "method",
        "latent_dim",
        "hidden_dim",
        "seed",
    ]

    if not all(
        column in df.columns
        for column in required
    ):
        return pd.Series(
            False,
            index=df.index,
            dtype=bool,
        )

    return (
        (
            df["method"]
            .astype(str)
            .str.lower()
            == method
        )
        & (
            pd.to_numeric(
                df["latent_dim"],
                errors="coerce",
            )
            == latent_dim
        )
        & (
            pd.to_numeric(
                df["hidden_dim"],
                errors="coerce",
            )
            == hidden_dim
        )
        & (
            pd.to_numeric(
                df["seed"],
                errors="coerce",
            )
            == seed
        )
    )


def _configuration_is_complete(
    raw_df,
    *,
    method,
    latent_dim,
    hidden_dim,
    seed,
):
    """Check whether all gene-space OT metrics are available.

    Parameters
    ----------
    raw_df
        Existing raw benchmark results.
    method
        Representation method.
    latent_dim
        Dimensionality of the latent representation.
    hidden_dim
        Number of hidden units in the latent SDE.
    seed
        Random seed associated with the model configuration.

    Returns
    -------
    bool
        ``True`` if the configuration exists and all required metrics
        are present and non-missing.
    """
    mask = _configuration_mask(
        raw_df,
        method=method,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        seed=seed,
    )

    rows = raw_df.loc[
        mask
    ]

    if len(rows) == 0:
        return False

    for metric in GENE_OT_METRIC_COLUMNS:
        if metric not in rows.columns:
            return False

        if rows[metric].isna().all():
            return False

    return True


def run_gene_space_ot():
    """Run gene-space OT evaluation for all benchmark configurations.

    Each configuration is evaluated using L2, cosine, and Hellinger
    ground costs. Every configuration runs in an isolated process so
    that memory allocated by the worker is released when the process
    terminates.

    Returns
    -------
    tuple of pandas.DataFrame
        Raw benchmark results and the aggregated summary dataframe.
    """
    benchmark_csv = (
        BENCHMARK_CSV
    )

    output_dir = (
        OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not benchmark_csv.exists():
        raise FileNotFoundError(
            "Benchmark CSV does not exist:\n"
            f"{benchmark_csv}"
        )

    benchmark = pd.read_csv(
        benchmark_csv
    )

    print(
        f"PROJECT_ROOT = "
        f"{PROJECT_ROOT}"
    )

    print(
        f"BENCHMARK = "
        f"{benchmark_csv}"
    )

    print(
        f"OUTPUT = "
        f"{output_dir}"
    )

    print(
        f"Benchmark rows = "
        f"{len(benchmark)}"
    )

    print(
        "Gene-space OT distances = "
        f"{GENE_OT_DISTANCES}"
    )

    raw_path = (
        output_dir
        / "results.csv"
    )

    if raw_path.exists():
        print(
            f"\nExisting results found:\n"
            f"{raw_path}"
        )

        raw_df = pd.read_csv(
            raw_path
        )

        print(
            "Existing rows = "
            f"{len(raw_df)}"
        )

    else:
        print(
            "\nNo existing results found."
        )

        raw_df = pd.DataFrame()

    with tempfile.TemporaryDirectory(
        prefix="gene_space_ot_"
    ) as tmp:
        tmp_dir = Path(
            tmp
        )

        for iteration, row in benchmark.iterrows():
            method = str(
                row["method"]
            ).lower()

            latent_dim = int(
                row["latent_dim"]
            )

            hidden_dim = int(
                row["hidden_dim"]
            )

            seed = int(
                row["seed"]
            )

            print()
            print(
                "#" * 80
            )

            print(
                f"ITERATION "
                f"{iteration + 1} / "
                f"{len(benchmark)}"
            )

            print(
                f"method      = {method}"
            )

            print(
                f"latent_dim  = {latent_dim}"
            )

            print(
                f"hidden_dim  = {hidden_dim}"
            )

            print(
                f"seed        = {seed}"
            )

            print(
                "#" * 80
            )

            if _configuration_is_complete(
                raw_df,
                method=method,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                seed=seed,
            ):
                print(
                    "SKIPPING: all L2 / cosine / "
                    "Hellinger metrics already exist.",
                    flush=True,
                )

                continue

            if len(raw_df) > 0:
                mask = _configuration_mask(
                    raw_df,
                    method=method,
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    seed=seed,
                )

                if mask.any():
                    print(
                        "Removing incomplete legacy "
                        "result for this configuration."
                    )

                    raw_df = (
                        raw_df.loc[
                            ~mask
                        ]
                        .reset_index(
                            drop=True
                        )
                    )

                    raw_df.to_csv(
                        raw_path,
                        index=False,
                    )

            print_ram(
                "before child"
            )

            output_file = (
                tmp_dir
                / f"result_{iteration}.json"
            )

            result = _run_isolated_iteration(
                method=method,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                seed=seed,
                output_file=output_file,
            )

            result.update(
                {
                    "method": method,
                    "latent_dim": latent_dim,
                    "hidden_dim": hidden_dim,
                    "seed": seed,
                }
            )

            result_df = pd.DataFrame(
                [result]
            )

            raw_df = pd.concat(
                [
                    raw_df,
                    result_df,
                ],
                ignore_index=True,
            )

            raw_df.to_csv(
                raw_path,
                index=False,
            )

            print(
                f"\nResult saved to:\n"
                f"{raw_path}",
                flush=True,
            )

            print_ram(
                "after child"
            )

            try:
                output_file.unlink()
            except FileNotFoundError:
                pass

            del result
            del result_df
            del output_file

            gc.collect()

    raw_df.to_csv(
        raw_path,
        index=False,
    )

    required_columns = [
        "method",
        "latent_dim",
        "hidden_dim",
        "seed",
        *GENE_OT_METRIC_COLUMNS,
    ]

    missing = [
        column
        for column in required_columns
        if column not in raw_df.columns
    ]

    if missing:
        raise RuntimeError(
            "Raw results are missing required columns:\n"
            f"{missing}"
        )

    configuration_columns = [
        "method",
        "latent_dim",
        "hidden_dim",
    ]

    seed_counts = (
        raw_df
        .groupby(
            configuration_columns
        )["seed"]
        .nunique()
    )

    invalid_configurations = (
        seed_counts[
            seed_counts != 3
        ]
    )

    if len(invalid_configurations) > 0:
        raise RuntimeError(
            "Each configuration must contain "
            "exactly 3 seeds.\n\n"
            f"{invalid_configurations}"
        )

    summary_metrics = list(
        GENE_OT_METRIC_COLUMNS
    )

    legacy_metrics = [
        "gene_ot_t1",
        "gene_ot_t2",
        "gene_baseline_ot_t1",
        "gene_baseline_ot_t2",
        "normalized_gene_ot_t1",
        "normalized_gene_ot_t2",
    ]

    summary_metrics.extend(
        [
            metric
            for metric in legacy_metrics
            if metric in raw_df.columns
        ]
    )

    available = [
        metric
        for metric in summary_metrics
        if metric in raw_df.columns
    ]

    if len(available) > 0:
        summary_df = (
            raw_df
            .groupby(
                configuration_columns
            )[available]
            .agg(
                [
                    "mean",
                    "std",
                ]
            )
        )

        summary_df.columns = [
            f"{metric}_{stat}"
            for metric, stat in summary_df.columns
        ]

        summary_df = (
            summary_df
            .reset_index()
        )

    else:
        summary_df = pd.DataFrame()

    summary_path = (
        output_dir
        / "summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"\nSummary written to:\n"
        f"{summary_path}"
    )

    if len(summary_df) > 0:
        make_plots(
            summary_df=summary_df,
            output_dir=output_dir,
        )

    return (
        raw_df,
        summary_df,
    )