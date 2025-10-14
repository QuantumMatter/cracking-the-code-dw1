# Neural Encoding Analysis

fMRI data and feature matrices for analyzing neural representations of object stimuli.

## Dataset Overview

**81 object stimuli** (airplane, armchair, bathtub, etc.) with:
- Neural responses from 7 brain regions (696 voxels total)
- AlexNet features (20-dimensional)
- Word2Vec features (30-dimensional)

All datasets are **index-aligned**: row `i` in each file corresponds to the same stimulus.

## Data Files

### 1. Neural Data (`python_roi_data.pickle`)

Dictionary with 4 keys:

- **`betas`** (81 × 696): Neural responses - rows=stimuli, columns=voxels
- **`conds`** (81 × 1): Stimulus labels ('airplane', 'chair', etc.)
- **`indices`** (1 × 696): Maps each voxel to its ROI (values 1-7)
- **`rois`** (1 × 7): ROI names: `['EVC', 'OPA', 'PPA', 'LOC', 'PFS', 'OFA', 'FFA']`

**Brain Regions:**
- **EVC**: Early Visual Cortex (100 voxels)
- **OPA**: Occipital Place Area (100 voxels)
- **PPA**: Parahippocampal Place Area (100 voxels)
- **LOC**: Lateral Occipital Complex (96 voxels)
- **PFS**: Parietal Face-Selective (100 voxels)
- **OFA**: Occipital Face Area (100 voxels)
- **FFA**: Fusiform Face Area (100 voxels)

### 2. Feature Matrices

- **`alexnet.mat`**: 81 × 20 AlexNet features
- **`word2vec.mat`**: 81 × 30 Word2Vec features

## Usage

### Load Neural Data

```python
from load_neural_data import load_neural_data, get_roi_data

# Load all data
data = load_neural_data()

# Get specific ROI
ffa_responses = get_roi_data(data, 'FFA')  # Shape: (81, 100)

# Access specific stimulus
airplane_response = data['betas'][0, :]  # All voxels for "airplane"
stimulus_name = data['conds'][0]  # 'airplane'
```

### Load Feature Matrices

```python
from load_feature_matrices import analyze_features

alexnet, word2vec = analyze_features()
# alexnet: (81, 20)
# word2vec: (81, 30)
```

### Example: Get FFA response to "airplane"

```python
data = load_neural_data()

# Method 1: Using helper function
ffa_data = get_roi_data(data, 'FFA')
airplane_ffa = ffa_data[0, :]  # 100 FFA voxels

# Method 2: Manual indexing
ffa_mask = data['indices'].flatten() == 7
airplane_ffa = data['betas'][0, ffa_mask]
```

## Data Structure

```
conds[0] = 'airplane'  ←→  betas[0, :] = all voxel responses to airplane
conds[1] = 'armchair'  ←→  betas[1, :] = all voxel responses to armchair
...
conds[80] = 'vase'     ←→  betas[80, :] = all voxel responses to vase
```

**Key Point**: Index `i` is consistent across all datasets:
- `conds[i]` = stimulus name
- `betas[i, :]` = neural response
- `alexnet[i, :]` = AlexNet features
- `word2vec[i, :]` = Word2Vec features

## Verification

Run `verify_data.py` to check data integrity:

```bash
uv run python verify_data.py
```

## Requirements

```bash
uv add scipy numpy
```
