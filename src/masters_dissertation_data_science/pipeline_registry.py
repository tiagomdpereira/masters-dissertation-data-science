"""Project pipelines."""

from kedro.pipeline import Pipeline

from masters_dissertation_data_science.pipelines.training.pipeline import (
    create_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""

    training_pipeline = create_pipeline()

    return {
        "__default__": training_pipeline,
        "training": training_pipeline,
    }
