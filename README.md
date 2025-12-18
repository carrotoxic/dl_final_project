# Learning Latent Play-Style Representations

This repository implements deep autoencoder architectures (LSTM and Transformer) to learn compact latent representations of gameplay trajectories for unsupervised play-style clustering in Super Mario Bros.

## Overview

The project extracts event-based temporal features (jumps, kills, collected items, deaths) from gameplay trajectories and uses autoencoders to compress variable-length sequences into fixed-size latent vectors. These representations are then clustered to identify distinct play-style archetypes.

## Installation

```bash
pip install -r requirements.txt
```

## Data Structure

Place trajectory data in `data/player_trajectory_cumulative_no_timeout_human/` as JSON files. Each file should contain a `trace` field with timestep event counters.

## Training Autoencoders

Train LSTM or Transformer autoencoder models:

```bash
# Train LSTM autoencoder
python -m player_model.auto_encoder.train --model lstm

# Train Transformer autoencoder
python -m player_model.auto_encoder.train --model transformer
```

Models are saved to `checkpoints/{model}/{model}.pt` with loss curves saved as PNG.

### Model Weights

Please download the pre-trained model weights from:
- 🤗 [Hugging Face](https://huggingface.co/carrotoxic/dl_midterm)

## Clustering

Run clustering analysis on learned latent representations:

```bash
# Run all clustering methods for all k values (2-10)
python -m player_model.clustering.main --model lstm

# Run specific method and cluster number
python -m player_model.clustering.main --model lstm --methods kmeans --n_clusters 5

# Run multiple methods
python -m player_model.clustering.main --model transformer --methods kmeans gmm spectral --n_clusters 5
```

Available clustering methods: `kmeans`, `gmm`, `agglomerative`, `dbscan`, `spectral`

Results are saved to `clustering_results/{model}/{method}/k{n_clusters}/` including:
- Cluster assignments and statistics
- 2D/3D PCA visualizations
- Silhouette scores
- Evaluation summaries

## Project Structure

```
player_model/
├── auto_encoder/          # Autoencoder models and training
│   ├── models/           # LSTM and Transformer architectures
│   ├── datasets/         # Data loading and preprocessing
│   └── train.py          # Training script
└── clustering/           # Clustering analysis
    ├── core/             # Clustering methods and evaluation
    ├── visualization/    # Plotting functions
    └── main.py           # Main clustering script
```

## Key Features

- Event-based feature extraction (12 features per timestep)
- Variable-length sequence handling with masking
- Multiple clustering algorithms with evaluation metrics
- Comprehensive visualizations (2D/3D projections, cluster statistics)
- Support for both LSTM and Transformer architectures
