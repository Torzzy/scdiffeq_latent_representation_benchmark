from scdiffeq.analysis.benchmark.normalize import (
    add_normalized_ot_to_benchmark,
)
from scdiffeq.analysis.benchmark.plots import (
    make_plots,
)
from scdiffeq.analysis.benchmark.summary import (
    load_and_summarize,
)
from scdiffeq.analysis.benchmark.tables import (
    make_tables,
)


def run_benchmark_analysis(paths):
    """Run the complete benchmark analysis pipeline.

    Parameters
    ----------
    paths
        Analysis paths containing the benchmark data, models, processed
        datasets, and output directories.

    Returns
    -------
    pandas.DataFrame
        Benchmark dataframe containing one row per seed.
    """
    add_normalized_ot_to_benchmark(
        benchmark_csv=paths.benchmark_csv,
        models_dir=paths.models,
        processed_dir=paths.processed,
    )

    raw_df, summary_df = load_and_summarize(
        paths.benchmark_csv,
    )

    benchmark_dir = paths.analysis / "benchmark"

    benchmark_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    make_tables(
        raw_df=raw_df,
        summary_df=summary_df,
        output_dir=benchmark_dir,
    )

    make_plots(
        summary_df=summary_df,
        output_dir=benchmark_dir,
    )

    return raw_df

if __name__ == "__main__":
    from scdiffeq.analysis.analysis_paths import AnalysisPaths
    paths = AnalysisPaths(
        root="../../../results",
    )
    run_benchmark_analysis(paths)