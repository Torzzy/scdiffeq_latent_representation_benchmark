from scdiffeq.analysis.analysis_paths import (
    AnalysisPaths,
)
from scdiffeq.analysis.benchmark.run import (
    run_benchmark_analysis,
)
from scdiffeq.analysis.clone_prediction.run import (
    run_clone_prediction_analysis,
)
from scdiffeq.analysis.gene_space_ot.run import (
    run_gene_space_ot,
)
from scdiffeq.analysis.metric_tensor.run import (
    run_metric_tensor_analysis,
)
from scdiffeq.analysis.vector_fields.run import (
    run_vector_field_analysis,
)

from scdiffeq.analysis.reconstruction_evaluation.run import (
    run_reconstruction_evaluation_analysis,
)

from scdiffeq.analysis.reconstruction_future_prediction_correlation import compute_reconstruction_prediction_correlations


def main():
    """Run the complete analysis pipeline."""
    paths = AnalysisPaths(
        root="../../results",
    )

    run_gene_space_ot()

    raw_df = run_benchmark_analysis(
        paths,
    )

    run_clone_prediction_analysis(
        raw_df=raw_df,
        paths=paths,
    )

    run_metric_tensor_analysis(
        raw_df=raw_df,
        paths=paths,
    )

    run_vector_field_analysis(
        raw_df=raw_df,
        paths=paths,
    )

    run_reconstruction_evaluation_analysis()

    compute_reconstruction_prediction_correlations()

    print()
    print("Analysis complete.")
    print(
        f"Results written to {paths.analysis}"
    )


if __name__ == "__main__":
    main()