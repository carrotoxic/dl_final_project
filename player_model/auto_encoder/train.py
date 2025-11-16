import os

import torch
from torch.utils.data import DataLoader
from torch import optim

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

    valid_elems = (lengths * D).sum().clamp(min=1)
    loss = diff.sum() / valid_elems
    return loss


def main():
    os.makedirs(train_config.model_save_path.parent, exist_ok=True)

    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")

    dataset = TrajectoryDataset(data_config)
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
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

    global_step = 0
    running_loss = 0.0

    for epoch in range(1, train_config.num_epochs + 1):
        model.train()

        for batch_idx, batch in enumerate(dataloader):
            trajectories = batch["trajectories"].to(device)
            lengths = batch["lengths"].to(device)

            optimizer.zero_grad()
            recon, _ = model(trajectories, lengths)
            loss = masked_mse_loss(recon, trajectories, lengths)
            loss.backward()
            optimizer.step()

            global_step += 1
            running_loss += loss.item()

            if global_step % train_config.log_interval == 0:
                avg_loss = running_loss / train_config.log_interval
                print(
                    f"Epoch {epoch} | Step {global_step} | "
                    f"Batch {batch_idx} | Loss {avg_loss:.6f}"
                )
                running_loss = 0.0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "data_config": data_config,
                "train_config": train_config,
                "epoch": epoch,
            },
            train_config.model_save_path,
        )
        print(f"[Epoch {epoch}] model saved to {train_config.model_save_path}")


if __name__ == "__main__":
    main()