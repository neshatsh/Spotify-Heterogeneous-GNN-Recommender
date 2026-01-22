# Spotify Heterogeneous Graph Neural Network Recommender
A graph-based music recommendation system that extends the Stanford CS224W bipartite playlist-track model into a heterogeneous Playlist–Track–Artist graph.
This project replicates and substantially improves upon the original ["Spotify Track Neural Recommender System"](https://medium.com/stanford-cs224w/spotify-track-neural-recommender-system-51d266e31e16) article with enhanced architectures, robust evaluation, and a practical recommendation engine.

## Project Overview
Traditional recommender systems treat playlists as simple lists. This project models music consumption as an interconnected graph where Graph Neural Networks (GNNs) can propagate information across shared connections, enabling more organic music discovery.

**Key Innovation: Heterogeneous Graph Structure**

Unlike the original bipartite approach, this system models three entity types:

- Playlists (22,083 nodes) - User-created collections
- Tracks (12,237 nodes) - Individual songs
- Artists (3,053 nodes) - Track performers

This richer structure allows the GNN to learn higher-level semantic relationships through artist metadata.

## Major Extensions Beyond Original Work
1. **Enhanced Graph Construction**

- Heterogeneous three-layer graph (Playlist–Track–Artist)
- Artist-preserving k-core filtering (prevents loss of metadata)
- Separate interaction edges (Playlist→Track) and structural edges (Track→Artist)

2. **Expanded Model Suite**

    Trained and evaluated 10 models across 5 architectures:

- LightGCN (LGC) - Simplified propagation baseline
- GraphSAGE - Neighbor sampling aggregation
- Graph Attention Network (GAT) - Attention-weighted messages
- Vanilla GCN - Spectral graph convolution (Kipf & Welling)
- Relational GCN (RGCN) - Relation-specific transformations for heterogeneous graphs

    Each architecture tested with both random and hard negative sampling strategies.
3. **Robust Training & Evaluation**

- Hard negative mining - Samples from highest-scoring false positives
- Track-safe negative sampling - Prevents sampling Artist nodes as negatives
- Enhanced metrics:

    - Recall@K (top-k recommendation accuracy)
    - ROC-AUC (global ranking quality)
    - Novelty score (measures non-mainstream recommendations)



4. **Advanced Visualizations**

- Snowball sampling for coherent subgraph extraction
- Multipartite hierarchical layouts showing all three layers
- Egocentric artist neighborhoods (Artist → Tracks → Playlists)

5. **Practical Recommendation Engine**

- **Playlist completion** - Suggest tracks for existing playlists
- **Cold-start "vibe check"** - Generate recommendations from 2-3 seed songs using centroid-based inference
- **Popularity re-ranking** - Reduce mainstream bias with degree-based penalties

## Key Results
**Winner: Vanilla GCN with Random Negative Sampling**

Unlike the original article where GraphSAGE dominated, the heterogeneous graph favors Vanilla GCN:

- Test AUC: 0.72 (highest)
- Recall@300: 0.163 (highest)
- Insight: Expressive message-passing with learnable weights outperforms simplified aggregation (LightGCN) once artist metadata is included

| Model | Sampling | Test AUC | Test Recall@300 | Test Novelty |
|-------|----------|----------|-----------------|--------------|
| **Vanilla GCN** | **random** | **0.721779** | **0.163129** | **11.098997** |
| GraphSAGE | random | 0.685859 | 0.141792 | 11.313659 |
| RGCN | random | 0.684087 | 0.134198 | 11.351497 |
| Vanilla GCN | hard | 0.605715 | 0.074027 | 11.819593 |
| RGCN | hard | 0.577826 | 0.050027 | 12.055986 |
| LightGCN | random | 0.575838 | 0.058243 | 12.407705 |
| GraphSAGE | hard | 0.548860 | 0.044118 | 12.310387 |
| LightGCN | hard | 0.536760 | 0.034934 | 12.653636 |
| GAT | random | 0.535298 | 0.034503 | 12.465388 |
| GAT | hard | 0.499546 | 0.024890 | 12.868133 |


## Key Findings

**Vanilla GCN + Random Sampling achieves best performance** (AUC: 0.72, Recall@300: 0.163)
- Differs from original article where GraphSAGE dominated
- Heterogeneous graphs favor expressive message-passing over simplified aggregation

**Key Patterns:**
- **RGCN** shows higher novelty (recommends less popular tracks via Artist relations)
- **Hard negatives** improve AUC but reduce Recall@300 (sharper boundaries, less generalization)
- **LightGCN** underperforms once Artists are added (simplified aggregation insufficient)
- **GAT** performs worst overall (attention unstable on large sparse graphs)

**Insight:** Adding Artist nodes fundamentally changes model rankings. The heterogeneous structure rewards learnable weights (GCN, RGCN) over parameter-free methods (LightGCN) and stable message-passing over attention (GAT).

## Installation
**Requirements**
```bash
Python 3.10+
PyTorch 2.1+
PyTorch Geometric 2.4+
CUDA 11.8+ (for GPU acceleration)
```

## Setup
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-scatter torch-sparse
pip install pandas numpy matplotlib networkx scikit-learn
```

## Data Acquisition
This project uses the Spotify Million Playlist Dataset (MPD).
Steps to reproduce:

1. Register at [AIcrowd](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)
2. Download the full dataset (~34 GB)
3. Extract the first 50 JSON files (slices 0–49)
4. Compress into spotify-data-50f.zip
5. Place in project root directory

Dataset stats (after 30-core filtering):

- 37,373 total nodes
- 1,555,870 edges
- ~2M unique tracks, ~300K artists in raw data

