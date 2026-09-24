import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from typing import Dict, Callable, Tuple
import logging

from .model import SingleCategoryDataset, SingleClassWGAN

logger = logging.getLogger(__name__)

def prepare_dataloader(
    time_series_data: Dict[str, Callable],
    df: pd.DataFrame,
    domain: str,
    anomaly: str,
    seq_len: int,
    batch_size: int,
) -> Tuple[DataLoader, SingleCategoryDataset]:
    
    dataset = SingleCategoryDataset(
        df=df,
        time_series_dict=time_series_data,
        domain=domain,
        anomaly=anomaly,
        seq_len=seq_len,
        splits=["train", "val"]
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    logger.info(f"Total generated samples using sliding window for training: {len(dataset)}")
    
    return dataloader, dataset


def build_model(
    seq_len: int,
    feature_dim: int,
    latent_dim: int,
    learning_rate: float,
    device: str,
) -> SingleClassWGAN:
    
    return SingleClassWGAN(
        seq_len=seq_len,
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        device=device
    )


def train_model(
    model: SingleClassWGAN,
    dataloader: DataLoader,
    epochs: int,
) -> Tuple[dict, dict]:
    
    model.G.train()
    model.D.train()

    history = {"d_loss": [], "d_real": [], "d_fake": [], "g_loss": []}
    global_step = 0 

    for epoch in range(epochs):
        epoch_d_loss, epoch_g_loss = [], []
        
        for real_data in dataloader:
            real_data = real_data.to(model.device)
            metrics = model.train_step(real_data)

            mlflow.log_metrics(metrics, step=global_step)
            global_step += 1

            epoch_d_loss.append(metrics["d_loss"])
            epoch_g_loss.append(metrics["g_loss"])

        history["d_loss"].append(np.mean(epoch_d_loss))
        history["g_loss"].append(np.mean(epoch_g_loss))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}] | D Loss: {np.mean(epoch_d_loss):.4f} | G Loss: {np.mean(epoch_g_loss):.4f}")

    weights = {
        "G_state_dict": model.G.state_dict(),
        "D_state_dict": model.D.state_dict(),
    }
    return weights, history


def generate_evaluation_plot(
    base_model: SingleClassWGAN,
    trained_state_dict: dict,
    dataset: SingleCategoryDataset,
    domain: str,
    anomaly: str,
    seq_len: int,
    target_length: int,
    feature_dim: int,
) -> plt.Figure:
    
    base_model.G.load_state_dict(trained_state_dict["G_state_dict"])
    base_model.G.eval()

    with torch.no_grad():
        num_chunks = target_length // seq_len  
        z_test = torch.randn(num_chunks, base_model.latent_dim, device=base_model.device)
        chunks_sinteticos = base_model.G(z_test).cpu().numpy().transpose(0, 2, 1) 
        generated_signal = chunks_sinteticos.reshape(-1, feature_dim) 
    generated_signal_denorm = (generated_signal * dataset.std) + dataset.mean

    file_name_real = dataset.df.loc[0, "ism330dhcx_acc"]
    ts_real_completo = dataset.ts_dict[file_name_real]()[["A_x [g]", "A_y [g]", "A_z [g]"]].values[1:target_length+1, :]

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f"Real vs Synthetic ({target_length} points) | {domain} | {anomaly}", fontsize=14, fontweight='bold')

    titles = ["Acceleration X [g]", "Acceleration Y [g]", "Acceleration Z [g]"]
    colors_real = ['royalblue', 'forestgreen', 'firebrick']
    colors_fake = ['darkorange', 'darkorange', 'darkorange']

    for i in range(feature_dim):
        axes[i].plot(ts_real_completo[:, i], label="Real", color=colors_real[i], alpha=0.9, linewidth=0.5)
        axes[i].plot(generated_signal_denorm[:, i], label="Synthetic", color=colors_fake[i], alpha=0.8, linewidth=0.5)
        axes[i].set_ylabel(titles[i], fontweight='bold')
        axes[i].legend(loc="upper right")

    axes[-1].set_xlabel("Timesteps", fontweight='bold')
    plt.tight_layout()
    
    return fig