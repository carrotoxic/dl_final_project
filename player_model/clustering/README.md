# Player Trajectory Clustering

This module provides tools for training a Transformer-based autoencoder on player trajectories and performing clustering analysis.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Commands

### 1. Train Autoencoder

Train the Transformer autoencoder on player trajectories:

```bash
python -m player_model.auto_encoder.train
```

The model will be saved to `checkpoints/player_model/auto_encoder/lstm_ae.pt`.

### 2. Player Clustering

Cluster players using latent representations and game statistics. Results are saved in separate folders for each cluster number.

**Loop through cluster numbers 2-12 (default):**
```bash
python -m player_model.player_clustering --mode combined
python -m player_model.player_clustering --mode latent_only
```

**Single cluster number:**
```bash
python -m player_model.player_clustering --n_clusters 5 --mode combined
```

**Filter out TIME_OUT traces:**
```bash
python -m player_model.player_clustering --mode combined --no_time_out
```

**Options:**
- `--n_clusters`: Number of clusters (default: loops through 2-12)
- `--mode`: `latent_only` or `combined` (default: `combined`)
- `--output_dir`: Output directory (default: `clusters`)
- `--no_time_out`: Filter out TIME_OUT traces

**Output structure:**
```
clusters/
├── cluster_num_2/
│   └── [mode]/
│       ├── trajectory_clusters_k2.json
│       └── player_clusters_[mode]_k2.csv
├── cluster_num_3/
│   └── ...
└── ...
```

### 3. Clustering Visualization

Generate PCA visualizations and plots for clustering results:

**Loop through cluster numbers 2-12 (default):**
```bash
python -m player_model.player_clustering_visualization --mode combined
python -m player_model.player_clustering_visualization --mode latent_only
```

**Single cluster number:**
```bash
python -m player_model.player_clustering_visualization --n_clusters 5 --mode combined
```

**Options:**
- `--n_clusters`: Number of clusters (default: loops through 2-12)
- `--mode`: `latent_only` or `combined` (default: `combined`)
- `--output_dir`: Output directory (default: `clusters`)
- `--no_timeout`: Filter out TIME_OUT traces

**Output:** JSON files, numpy arrays, and PCA visualization plots (2D/3D) for clusters, players, and player types.

### 4. Cluster Statistics

Display statistics for clustering results:

```bash
python -m player_model.cluster_statistics --csv_file clusters/cluster_num_4/player_clusters_combined_k4.csv
```

**Options:**
- `--csv_file`: Path to CSV file with cluster assignments (required)
- `--json_file`: Path to trajectory-level JSON file (optional, auto-detected if not provided)

### 5. DTW Clustering

Cluster trajectories using Dynamic Time Warping (DTW) distance:

```bash
python -m player_model.dtw_clustering --n_clusters 8 --output_dir clusters_dtw
```

**Options:**
- `--n_clusters`: Number of clusters (default: 8)
- `--output_dir`: Output directory (default: `clusters_dtw_fast`)
- `--max_files`: Maximum number of files to process (optional)

## Architecture

The autoencoder uses a **LSTM-based sequence encoder** with:
- **Encoder**: LSTM encoder (unidirectional)
- **Decoder**: Non-autoregressive MLP decoder
- **Loss**: Mean Euclidean distance in original (denormalized) space

## Data Format

Trajectory data should be in `data/player_trajectory/` as JSON files with:
- `trace`: List of [x, y] coordinates
- `completing-ratio`, `#kills`, `#coins`, `lives`, `all_enemies`, `all_coins`, etc.

