from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    prepare_dataloader,
    build_model,
    train_model,
    generate_evaluation_plot,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # Step 1: Prepare Dataset
            node(
                func=prepare_dataloader,
                inputs=[
                    "time_series_folder",  
                    "df_splitted",  
                    "params:domain_shift_op",
                    "params:anomaly_label",
                    "params:seq_len",
                    "params:batch_size",
                ],
                outputs=["dataloader", "dataset"],
                name="prepare_dataloader_node",
            ),
            # Step 2: Build Model Architecture
            node(
                func=build_model,
                inputs=[
                    "params:seq_len",
                    "params:feature_dim",
                    "params:latent_dim",
                    "params:learning_rate",
                    "params:device",
                ],
                outputs="model",
                name="build_model_node",
            ),
            # Step 3: Run Training Loop
            node(
                func=train_model,
                inputs=[
                    "model",
                    "dataloader",
                    "params:epochs",
                ],
                outputs=["trained_model_weights", "training_history"],
                name="train_model_node",
            ),
            # Step 4: Generate Inference and Plot
            node(
                func=generate_evaluation_plot,
                inputs=[
                    "model",
                    "trained_model_weights",
                    "dataset",
                    "params:domain_shift_op",
                    "params:anomaly_label",
                    "params:seq_len",
                    "params:target_length",
                    "params:feature_dim",
                ],
                outputs="evaluation_plots",
                name="evaluation_plots_node",
            ),
        ],
        namespace="training",
        inputs=None,
        outputs=None,
    )