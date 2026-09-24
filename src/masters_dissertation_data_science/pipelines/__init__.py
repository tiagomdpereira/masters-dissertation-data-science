from masters_dissertation_data_science.pipelines.training import (
    pipeline as training_pipeline,
)


def register_pipelines():
    """Registers all project pipelines."""
    return {
        "training": training_pipeline.create_pipeline(),
        "__default__": training_pipeline.create_pipeline(),
    }
