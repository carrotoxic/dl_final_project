import os

import torch
from torch.utils.data import DataLoader, random_split, Subset
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
from .utils.losses import masked_mse_loss
from .utils.metrics import evaluate, compute_training_metrics
from .utils.plot import plot_train_val_loss


def main():
    os.makedirs(train_config.model_save_path.parent, exist_ok=True)

    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")

    # Load full dataset with per-trajectory normalization to [0, 1]
    full_dataset = TrajectoryDataset(data_config, normalize=True)
    print(f"Loaded {len(full_dataset)} total samples")
    print(f"Using per-trajectory min-max normalization to [0, 1] per dimension")
    
    # Split into train (90%) and validation (10%)
    train_size = int(1.0 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config.num_epochs,
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
            
            if train_config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_config.gradient_clip
                )
            
            optimizer.step()
            scheduler.step()

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
        
        # Compute training metrics in original space
        train_euclidean_dist = compute_training_metrics(model, train_dataloader, device)
        
        # Evaluate on validation set
        val_results = evaluate(model, val_dataloader, device, compute_original_metrics=True)
        val_loss = val_results["normalized_loss"]
        val_euclidean_dist = val_results.get("mean_euclidean_distance", 0.0)
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
            f"Train Loss (norm): {epoch_train_avg:.6f} | "
            f"Train Euclidean Dist: {train_euclidean_dist:.4f} | "
            f"Val Loss (norm): {val_loss:.6f} {status} | "
            f"Val Euclidean Dist: {val_euclidean_dist:.4f} | "
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