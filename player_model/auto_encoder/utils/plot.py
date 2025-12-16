import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional


def plot_train_val_loss(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[Path] = None,
    show_plot: bool = True,
) -> None:
    """Plot training and validation loss curves"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

