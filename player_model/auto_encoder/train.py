import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
from tqdm import tqdm

from .config import (
    DataConfig,
    TrainConfig,
    create_model_config,
)
from .datasets.trajectory_datasets import TrajectoryDataset, trajectory_collate_fn
from .models import LSTMAutoencoder, TransformerAutoencoder, TrajectoryFeatureFusionAutoencoder
from .utils.losses import masked_mse_loss, fusion_reconstruction_loss
from .utils.plot import plot_train_val_loss


def create_model(model_config, device, dataset=None):
    """Create model based on config"""
    model_name = model_config.model_name
    if model_name == "lstm":
        return LSTMAutoencoder(model_config).to(device)
    elif model_name == "transformer":
        if dataset is None:
            raise ValueError("Transformer model requires dataset to infer max_seq_len")
        max_seq_len = dataset.max_seq_len
        return TransformerAutoencoder(model_config, max_seq_len).to(device)
    elif model_name == "fusion":
        return TrajectoryFeatureFusionAutoencoder(model_config).to(device)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def prepare_features(overall_features, target_dim=10):
    """Prepare features for fusion model (pad to target_dim if needed)."""
    if overall_features.shape[1] < target_dim:
        padding = torch.zeros(overall_features.shape[0], target_dim - overall_features.shape[1], device=overall_features.device)
        return torch.cat([overall_features, padding], dim=1)
    return overall_features[:, :target_dim]


def compute_loss(model, batch, model_name, model_config, train_config, device):
    """Compute loss based on model type."""
    trajectories = batch["trajectories"].to(device)
    lengths = batch["lengths"].to(device)
    
    if model_name == "fusion":
        overall_features = batch["overall_features"].to(device)
        raw_features = prepare_features(overall_features, model_config.raw_feature_dim)
        latent, recon_trajectories, recon_features = model.forward(
            trajectories=trajectories,
            lengths=lengths,
            raw_features=raw_features,
        )
        loss = fusion_reconstruction_loss(
            recon_trajectories=recon_trajectories,
            target_trajectory=trajectories,
            recon_raw=recon_features,
            target_raw=raw_features,
        )
        return loss, {"loss": loss.item()}
    else:
        latent, recon_trajectories = model.forward(trajectories, lengths)
        loss = masked_mse_loss(
            recon=recon_trajectories,
            target=trajectories,
            lengths=lengths,
        )
        return loss, {"loss": loss.item()}


def train_epoch(model, dataloader, model_name, model_config, train_config, device, optimizer, scheduler):
    """Train for one epoch."""
    model.train()
    epoch_losses = []
    epoch_components = []
    
    pbar = tqdm(dataloader, desc=f"Train", unit="batch")
    for batch in pbar:
        optimizer.zero_grad()
        loss, components = compute_loss(model, batch, model_name, model_config, train_config, device)
        loss.backward()
        
        if train_config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip)
        
        optimizer.step()
        scheduler.step()
        
        epoch_losses.append(loss.item())
        epoch_components.append(components)
        
        pbar.set_postfix({k: f"{v:.4f}" for k, v in components.items()})
    
    return np.mean(epoch_losses), epoch_components


def validate(model, dataloader, model_name, model_config, train_config, device):
    """Validate model."""
    model.eval()
    val_losses = []
    val_components = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val", leave=False):
            loss, components = compute_loss(model, batch, model_name, model_config, train_config, device)
            val_losses.append(loss.item())
            val_components.append(components)
    
    return np.mean(val_losses), val_components


def main():
    data_config = DataConfig()
    train_config = TrainConfig()
    model_config = create_model_config(model_name=train_config.model_name)
    
    os.makedirs(train_config.model_save_path.parent, exist_ok=True)
    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
    
    print(f"[INFO] Model: {train_config.model_name}")
    print(f"[INFO] Device: {device}")
    
    dataset = TrajectoryDataset(data_config)
    print(f"[INFO] Loaded {len(dataset)} samples")
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=trajectory_collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=trajectory_collate_fn,
        pin_memory=True,
    )
    
    model = create_model(model_config, device, dataset)
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config.num_epochs,
    )
    
    train_losses = []
    val_losses = []
    
    print(f"[INFO] Starting training for {train_config.num_epochs} epochs")
    
    for epoch in range(1, train_config.num_epochs + 1):
        train_loss, train_components = train_epoch(
            model, train_loader, train_config.model_name, model_config, train_config, device, optimizer, scheduler
        )
        val_loss, val_components = validate(
            model, val_loader, train_config.model_name, model_config, train_config, device
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if train_config.log_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": model_config,
                "train_config": train_config,
                "train_losses": train_losses,
                "val_losses": val_losses,
            }, train_config.model_save_path)
        
        avg_components = {
            k: np.mean([c[k] for c in train_components])
            for k in train_components[0].keys()
        }
        val_avg_components = {
            k: np.mean([c[k] for c in val_components])
            for k in val_components[0].keys()
        }
        
        print(
            f"Epoch {epoch}/{train_config.num_epochs} | "
            f"Train: {train_loss:.6f} {avg_components} | "
            f"Val: {val_loss:.6f} {val_avg_components}"
        )
    
    plot_path = train_config.model_save_path.parent / "loss_curves.png"
    plot_train_val_loss(
        train_losses=train_losses,
        val_losses=val_losses,
        save_path=plot_path,
        show_plot=False,
    )
    
    print(f"[INFO] Training complete. Model saved to: {train_config.model_save_path}")


if __name__ == "__main__":
    main()
