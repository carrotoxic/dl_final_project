import os

import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
from tqdm import tqdm

from .config import (
    data_config,
    model_config,
    train_config,
)
from .datasets.trajectory_datasets import (
    TrajectoryDataset,
    trajectory_collate_fn,
)
from .models.lstm_autoencoder import LSTMAutoencoder
from .utils.plot import plot_train_val_loss


def masked_mse_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    device = recon.device
    B, T_max, D = recon.shape
    lengths = lengths.to(device)

    mask = torch.arange(T_max, device=device).unsqueeze(0).expand(B, T_max)
    mask = (mask < lengths.unsqueeze(1)).float().unsqueeze(-1)

    diff = (recon - target) ** 2
    diff = diff * mask

    per_sample_sum = diff.sum(dim=(1, 2))

    valid_elems = (lengths * D).clamp(min=1)
    per_sample_mse = per_sample_sum / valid_elems
    loss = per_sample_mse.mean()
    return loss


def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            trajectories = batch["trajectories"].to(device)
            lengths = batch["lengths"].to(device)
            
            recon, _ = model(trajectories, lengths)
            loss = masked_mse_loss(recon, trajectories, lengths)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    os.makedirs(train_config.model_save_path.parent, exist_ok=True)

    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")

    # Load full dataset to compute normalization stats on all data
    full_dataset = TrajectoryDataset(data_config, normalize=True)
    print(f"Loaded {len(full_dataset)} total samples")
    if full_dataset.normalize:
        print(f"Normalization stats - Mean: {full_dataset.mean.numpy()}, Std: {full_dataset.std.numpy()}")
    
    # Split into train (90%) and validation (10%)
    train_size = int(1.0 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=trajectory_collate_fn,
        num_workers=1,
        pin_memory=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=trajectory_collate_fn,
        num_workers=1,
        pin_memory=True,
    )

    model = LSTMAutoencoder(model_config).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    # Track losses for plotting
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    global_step = 0

    for epoch in range(1, train_config.num_epochs + 1):
        model.train()
        epoch_train_losses = []
        running_loss = 0.0
        sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config.num_epochs)

        pbar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch}/{train_config.num_epochs}",
            unit="batch"
        )

        for batch_idx, batch in pbar:
            trajectories = batch["trajectories"].to(device)
            lengths = batch["lengths"].to(device)

            optimizer.zero_grad()
            recon, latent = model(trajectories, lengths)

            loss = masked_mse_loss(recon, trajectories, lengths)
            loss.backward()
            sched.step()
            
            if train_config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_config.gradient_clip
                )
            
            optimizer.step()

            global_step += 1
            loss_value = loss.item()
            epoch_train_losses.append(loss_value)
            running_loss += loss_value

            avg_loss = running_loss / (batch_idx + 1)
            current_epoch_avg = sum(epoch_train_losses) / len(epoch_train_losses)
            
            pbar.set_postfix({
                'loss': f'{loss_value:.6f}',
                'avg_loss': f'{avg_loss:.6f}',
                'epoch_avg': f'{current_epoch_avg:.6f}',
                'step': global_step
            })


        # Calculate average training loss for this epoch
        epoch_train_avg = sum(epoch_train_losses) / len(epoch_train_losses) if epoch_train_losses else 0.0
        train_losses.append(epoch_train_avg)
        
        # Evaluate on validation set
        val_loss = evaluate(model, val_dataloader, device)
        val_losses.append(val_loss)
        
        # Track best validation loss
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if epoch % train_config.save_interval == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "data_config": data_config,
                    "train_config": train_config,
                    "epoch": epoch,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "best_val_loss": best_val_loss,
                },
                train_config.model_save_path,
            )
            print(f"Model saved to {train_config.model_save_path}")
        
        status = "★ BEST" if is_best else ""
        print(
            f"\n[Epoch {epoch}/{train_config.num_epochs}] "
            f"Train Loss: {epoch_train_avg:.6f} | "
            f"Val Loss: {val_loss:.6f} {status} | "
            f"LR: {current_lr:.2e}"
        )
    
    # Plot training curves
    plot_path = train_config.model_save_path.parent / "loss_curves.png"
    plot_train_val_loss(
        train_losses=train_losses,
        val_losses=val_losses,
        save_path=plot_path,
        show_plot=False,
    )


if __name__ == "__main__":
    main()