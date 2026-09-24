import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Dict, Tuple

class SingleCategoryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, time_series_dict: Dict, domain: str, anomaly: str, seq_len: int, splits=["train", "val"], step_size=None):
        if isinstance(splits, str): splits = [splits]
            
        self.df = df[
            (df["domain_shift_op"] == domain) & 
            (df["anomaly_label"] == anomaly) & 
            (df["split"].isin(splits))
        ].reset_index(drop=True)
            
        if len(self.df) == 0:
            raise ValueError(f"Without samples for {domain} | {anomaly}")
            
        self.ts_dict = time_series_dict
        self.seq_len = seq_len
        self.step_size = step_size if step_size else seq_len // 2
        
        all_data = []
        for idx in range(len(self.df)):
            file_name = self.df.loc[idx, "ism330dhcx_acc"]
            ts = self.ts_dict[file_name]()[["A_x [g]", "A_y [g]", "A_z [g]"]].values[1:, :]
            
            for start_idx in range(0, len(ts) - self.seq_len + 1, self.step_size):
                window = ts[start_idx : start_idx + self.seq_len]
                all_data.append(window)
                
        all_data = np.array(all_data) 
        
        self.mean = np.mean(all_data, axis=(0, 1))
        self.std = np.std(all_data, axis=(0, 1)) + 1e-8
        self.data_list = (all_data - self.mean) / self.std

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return torch.tensor(self.data_list[idx], dtype=torch.float32).permute(1, 0)


class Generator1D(nn.Module):
    def __init__(self, latent_dim: int, seq_len: int, feature_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        self.init_len = seq_len // 4
        self.fc = nn.Linear(latent_dim, 64 * self.init_len)
        
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, feature_dim, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, self.init_len) 
        x = self.conv_blocks(x)
        if x.size(2) != self.seq_len:
            x = F.interpolate(x, size=self.seq_len)
        return x


class Discriminator1D(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(feature_dim, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.LazyLinear(1) 
        )

    def forward(self, x):
        return self.model(x).squeeze()


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True, retain_graph=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


class SingleClassWGAN:
    def __init__(self, seq_len: int, feature_dim: int, latent_dim: int, learning_rate: float, device: str):
        self.device = torch.device(device)
        self.latent_dim = latent_dim
        
        self.G = Generator1D(latent_dim, seq_len, feature_dim).to(self.device)
        self.D = Discriminator1D(feature_dim).to(self.device)
        
        _ = self.D(torch.randn(2, feature_dim, seq_len, device=self.device))
        
        self.g_opt = optim.Adam(self.G.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        self.d_opt = optim.Adam(self.D.parameters(), lr=learning_rate, betas=(0.5, 0.9))

    def train_step(self, real_data: torch.Tensor) -> Dict[str, float]:
        batch_size = real_data.size(0)

        for _ in range(5): 
            self.d_opt.zero_grad()
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_data = self.G(z).detach()

            d_real = self.D(real_data).mean()
            d_fake = self.D(fake_data).mean()
            gp = compute_gradient_penalty(self.D, real_data, fake_data, self.device)
            
            d_loss = d_fake - d_real + 10.0 * gp
            d_loss.backward()
            self.d_opt.step()

        self.g_opt.zero_grad()
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_data = self.G(z)
        
        g_loss = -self.D(fake_data).mean()
        
        g_loss.backward()
        self.g_opt.step()

        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "d_real": d_real.item(),
            "d_fake": d_fake.item()
        }