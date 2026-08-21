import contextlib
import os
import sys
import shutil
import csv
from pathlib import Path
import numpy as np

import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Add project src directory to system path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from diffusion.dataset import TrajectoryDataset
from diffusion.model import TemporalUnet
from diffusion.diffusion import GaussianDiffusion
from utils import abs_path, get_output_dir

log = logging.getLogger(__name__)


class EMAManager:
    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow_state = {
            k: v.clone().detach() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        current_state = model.state_dict()
        for k in self.shadow_state:
            if self.shadow_state[k].is_floating_point():
                self.shadow_state[k].mul_(self.decay).add_(
                    current_state[k], alpha=1.0 - self.decay
                )
            else:
                self.shadow_state[k].copy_(current_state[k])

    @contextlib.contextmanager
    def average_parameters(self, model: torch.nn.Module):
        raw_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow_state)
        try:
            yield
        finally:
            model.load_state_dict(raw_state)


def run_epoch(
    model: GaussianDiffusion,
    loader: DataLoader,
    optimizer=None,
    device: torch.device = torch.device("cpu"),
    n_cond_steps: int = 2,
    ema: EMAManager = None,
) -> float:
    """Single train or eval epoch. Returns mean loss over batches."""
    is_train = optimizer is not None
    model.train(is_train)
    running = 0.0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            tau = batch["tau"].to(device)  # (B, H, 2)
            r = batch["r"].to(device)  # (B, 1)
            b_mean = batch["b_mean"].to(device)  # (B, 1600)
            b_var = batch["b_var"].to(device)  # (B, 1600)

            cond = {}
            for cond_step in range(n_cond_steps):
                cond[cond_step] = tau[:, cond_step, :].clone()  # (B, 2)
                
            # cond = {0: tau[:, 0, :].clone()}

            loss, _ = model.loss(tau, cond, b_mean, b_var, r)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if ema is not None:
                    ema.update(model)

            running += loss.item()

    return running / len(loader)


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val_loss, ema: EMAManager = None, stats: dict = None):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }
    if ema is not None:
        state["ema_shadow"] = ema.shadow_state
    if stats is not None:
        state["stats"] = stats
    torch.save(state, path)


@hydra.main(
    config_path="../../config", config_name="train_diffusion", version_base="1.2"
)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = get_output_dir()
    log.info(f"Output directory: {output_dir}")

    dataset_dir = abs_path(cfg.dataset_dir)
    log.info(f"Loading dataset from: {dataset_dir}")

    train_path = os.path.join(dataset_dir, "training_data.pkl")
    val_path = os.path.join(dataset_dir, "validation_data.pkl")

    max_step = float(OmegaConf.select(cfg, "planner.max_step", default=10.0))
    initial_heading = float(OmegaConf.select(cfg, "planner.initial_heading", default=np.pi / 4))
    grid_size = int(np.round(np.sqrt(cfg.n_evaluations)))

    train_set = TrajectoryDataset(
        train_path,
        tuple(cfg.domain_size),
        cfg.domain_pad,
        horizon=cfg.horizon,
        stride=cfg.stride,
        reward_key=cfg.reward_key,
        reward_type=cfg.reward_type,
        crop_size=cfg.architecture.get("crop_size", 40),
        domain_min=list(cfg.domain_min),
        domain_max=list(cfg.domain_max),
        grid_size=grid_size,
        max_step=max_step,
        initial_heading=initial_heading,
        use_egocentric=cfg.architecture.get("use_egocentric", False),
    )
    val_set = TrajectoryDataset(
        val_path,
        tuple(cfg.domain_size),
        cfg.domain_pad,
        horizon=cfg.horizon,
        stride=cfg.stride,
        reward_key=cfg.reward_key,
        reward_type=cfg.reward_type,
        crop_size=cfg.architecture.get("crop_size", 40),
        domain_min=list(cfg.domain_min),
        domain_max=list(cfg.domain_max),
        grid_size=grid_size,
        max_step=max_step,
        initial_heading=initial_heading,
        use_egocentric=cfg.architecture.get("use_egocentric", False),
    )

    train_loader = DataLoader(train_set, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.training.batch_size)
    log.info(f"Training samples: {len(train_set)}  Validation samples: {len(val_set)}")

    backbone = TemporalUnet(
        horizon=cfg.architecture.horizon,
        transition_dim=cfg.architecture.transition_dim,
        belief_dim=cfg.architecture.belief_dim,
        dim=cfg.architecture.dim,
        dim_mults=tuple(cfg.architecture.dim_mults),
        condition_dropout=cfg.architecture.condition_dropout,
        kernel_size=cfg.architecture.kernel_size,
        crop_size=cfg.architecture.get("crop_size", 40),
        belief_encoder_pooling=cfg.architecture.get("belief_encoder_pooling", False),
        conditioning_type=cfg.architecture.get("conditioning_type", "cnn"),
    )
    model = GaussianDiffusion(
        backbone,
        horizon=cfg.diffusion.horizon,
        observation_dim=cfg.diffusion.observation_dim,
        n_timesteps=cfg.diffusion.n_timesteps,
        loss_type=cfg.diffusion.loss_type,
        clip_denoised=cfg.diffusion.clip_denoised,
        predict_epsilon=cfg.diffusion.predict_epsilon,
        loss_discount=cfg.diffusion.loss_discount,
        condition_guidance_w=cfg.diffusion.condition_guidance_w,
        n_cond_steps=cfg.diffusion.n_cond_steps,
        belief_guidance_w=cfg.diffusion.get("belief_guidance_w", None),
        return_guidance_w=cfg.diffusion.get("return_guidance_w", None),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_params:,}")

    ema = EMAManager(model, decay=cfg.training.ema_decay)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.n_epochs, eta_min=cfg.training.eta_min
    )

    log_path = os.path.join(output_dir, "training_log.csv")
    log_fields = ["epoch", "train_loss", "val_loss"]
    with open(log_path, mode="w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=log_fields)
        writer.writeheader()

    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(output_dir, "best_checkpoint.pt")

    for epoch in tqdm(range(1, cfg.training.n_epochs + 1), desc="Training"):
        train_loss = run_epoch(model, train_loader, optimizer, device, cfg.diffusion.n_cond_steps, ema)
        with ema.average_parameters(model):
            val_loss = run_epoch(model, val_loader, None, device, cfg.diffusion.n_cond_steps)
        tqdm.write(
            f"Epoch {epoch}/{cfg.training.n_epochs}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}"
        )
        scheduler.step()

        with open(log_path, mode="a", newline="") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=log_fields)
            writer.writerow(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                best_checkpoint_path, epoch, model, optimizer, scheduler, best_val_loss, ema, getattr(train_set, "stats", None)
            )

    final_checkpoint_path = os.path.join(output_dir, "final_checkpoint.pt")
    save_checkpoint(final_checkpoint_path, epoch, model, optimizer, scheduler, best_val_loss, ema, getattr(train_set, "stats", None))
    log.info(f"Best model saved with val loss: {best_val_loss:.4f}")
    log.info(f"Final model saved with val loss: {val_loss:.4f}")




if __name__ == "__main__":
    main()
