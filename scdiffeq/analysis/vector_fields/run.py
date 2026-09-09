from .evaluate import (
    evaluate_vector_field,
)


def run_vector_field_analysis(
    paths,
    raw_df,
):
    """
    Generate vector field visualizations for benchmark models.
    """

    ####################################################
    # output directory
    ####################################################

    output_dir = (
        paths.analysis
        / "vector_fields"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ####################################################
    # configurations
    ####################################################

    configs = (
        raw_df[
            [
                "method",
                "latent_dim",
                "hidden_dim",
                "seed",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "method",
                "latent_dim",
                "hidden_dim",
                "seed",
            ]
        )
    )

    ####################################################
    # loop
    ####################################################

    for _, cfg in configs.iterrows():

        method = cfg.method
        latent_dim = int(cfg.latent_dim)
        hidden_dim = int(cfg.hidden_dim)
        seed = int(cfg.seed)

        ################################################
        # keep one reference model
        ################################################

        if seed != 0:
            continue

        if hidden_dim != 256:
            continue

        ################################################
        # dataset
        ################################################

        dataset_path = paths.dataset(
            method,
            latent_dim,
        )

        if not dataset_path.exists():
            continue

        ################################################
        # checkpoint
        ################################################

        checkpoint = paths.checkpoint(
            method,
            latent_dim,
            hidden_dim,
            seed,
        )

        if not checkpoint.exists():
            continue

        print(
            f"[Vector field] "
            f"{method} "
            f"latent={latent_dim}"
        )

        ################################################
        # evaluate
        ################################################

        evaluate_vector_field(
            dataset_path=dataset_path,
            checkpoint_path=checkpoint,
            output_path=(
                output_dir
                / (
                    f"{method}"
                    f"_latent_{latent_dim}"
                    f"_hidden_{hidden_dim}"
                    f"_seed_{seed}.png"
                )
            ),
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            method=method,
        )