import math

# import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import gpytorch
from matplotlib import pyplot as plt
import joblib
import sys

from diffusion.model import TemporalUnet
from diffusion.diffusion import GaussianDiffusion
from diffusion.dataset import TrajectoryDataset


def run_epoch(model, dataloader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for batch in dataloader:
            tau = batch["tau"]  # (B, H, 2)
            r = batch["r"]  # (B, 1)
            b_mean = batch["b_mean"]  # (B, 1600)
            b_var = batch["b_var"]  # (B, 1600)

            # pin AUV start position as conditioning
            cond = {0: tau[:, 0, :].clone()}

            loss, info = model.loss(tau, cond, b_mean, b_var, r)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(dataloader)
    return epoch_loss



architecture = TemporalUnet(
    horizon=16,
    transition_dim=2,
    belief_dim=1600,
    dim=128,
    dim_mults=(1, 2, 4, 8),
    condition_dropout=0.1,
    kernel_size=5,
)


def train(model, train_loader, val_loader, n_epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(n_epochs):
        train_loss = run_epoch(model, train_loader, optimizer)
        val_loss = run_epoch(model, val_loader, None)
        print(
            f"Epoch {epoch + 1}/{n_epochs}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}"
        )


def main():
    path = "data/trajectories/processed/training_data.pkl"
    domain_size = (10.0, 10.0)
    dataset = TrajectoryDataset(path, domain_size)
    print(f"Loaded dataset with {len(dataset)} samples.")

    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    print(f"Training samples: {len(train_set)}  Validation samples: {len(val_set)}")

    diffusion_model = GaussianDiffusion(
        architecture, horizon=16, n_timesteps=100, loss_type="l2"
    )
    train(diffusion_model, train_loader, val_loader, n_epochs=10)


if __name__ == "__main__":
    main()
