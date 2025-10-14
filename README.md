# Neural Encoding Analysis

fMRI data from a **single subject** viewing images of objects. This dataset contains voxelwise responses from functionally defined ROIs to 81 categories of objects (e.g., airplane, boat, car, tree).

## Dataset Overview

**81 object categories** with mean responses to 10 images per category:
- Neural responses from 7 brain regions (696 voxels total)
- AlexNet features: 20 principal components from 5 convolutional layers
- Word2Vec features: 30 principal components

**Important**: The neural data (betas) are **mean voxelwise responses** across 10 images per category, derived from a GLM with a regressor for each object category.

All datasets are **index-aligned**: row `i` in each file corresponds to the same object category.

## Data Files

### 1. Neural Data (`python_roi_data.pickle`)

Dictionary with 4 keys:

- **`betas`** (81 × 696): Voxelwise responses to each condition (mean beta values from GLM)
  - Rows = object categories
  - Columns = voxels

- **`conds`** (81 × 1): Object category labels (e.g., 'airplane', 'boat', 'car')
  - Order corresponds to rows in `betas`

- **`indices`** (1 × 696): Voxel-to-ROI mapping (values 1-7)
  - Each value indicates which ROI that voxel belongs to
  - Example: `inds = indices == 2` selects all OPA voxels

- **`rois`** (1 × 7): ROI names in order
  - `['EVC', 'OPA', 'PPA', 'LOC', 'PFS', 'OFA', 'FFA']`

**Brain Regions (ROIs):**
- **EVC**: Early Visual Cortex (100 voxels)
- **OPA**: Occipital Place Area (100 voxels)
- **PPA**: Parahippocampal Place Area (100 voxels)
- **LOC**: Lateral Occipital Complex (96 voxels)
- **PFS**: Posterior Fusiform (100 voxels)
- **OFA**: Occipital Face Area (100 voxels)
- **FFA**: Fusiform Face Area (100 voxels)

### 2. Feature Matrices

- **`alexnet.mat`**: 81 × 20 principal components from AlexNet's 5 convolutional layers (trained on ImageNet)
- **`word2vec.mat`**: 81 × 30 principal components from word2vec embeddings

Both feature matrices contain **mean responses** to all 10 images per category.

## Usage

### Load Neural Data

```python
from load_neural_data import load_neural_data, get_roi_data

# Load all data
data = load_neural_data()

# Get specific ROI responses (all categories)
ffa_responses = get_roi_data(data, 'FFA')  # Shape: (81, 100)

# Access specific category response
airplane_response = data['betas'][0, :]  # All 696 voxels for "airplane" category
category_name = data['conds'][0]  # ['airplane']
```

### Load Feature Matrices

```python
from load_feature_matrices import analyze_features

alexnet, word2vec = analyze_features()
# alexnet: (81, 20) - mean AlexNet features per category
# word2vec: (81, 30) - mean Word2Vec features per category
```

### Example: Get FFA response to "airplane" category

```python
data = load_neural_data()

# Method 1: Using helper function
ffa_data = get_roi_data(data, 'FFA')
airplane_ffa = ffa_data[0, :]  # 100 FFA voxels, mean response to airplane

# Method 2: Manual indexing (equivalent)
ffa_mask = data['indices'].flatten() == 7
airplane_ffa = data['betas'][0, ffa_mask]
```

## Data Structure

All arrays are **index-aligned** by object category:

```
conds[0] = 'airplane'  ←→  betas[0, :] = mean voxel responses to airplane category (10 images)
conds[1] = 'armchair'  ←→  betas[1, :] = mean voxel responses to armchair category (10 images)
...
conds[80] = 'vase'     ←→  betas[80, :] = mean voxel responses to vase category (10 images)
```

**Key Point**: Index `i` is consistent across all datasets:
- `conds[i]` = object category name
- `betas[i, :]` = mean neural response (696 voxels)
- `alexnet[i, :]` = mean AlexNet features (20 PCs)
- `word2vec[i, :]` = mean Word2Vec features (30 PCs)

### Extracting ROI Data

To get data for a specific ROI, use the `indices` array:

```python
# Get all voxels for ROI 2 (OPA)
roi_idx = 2
inds = data['indices'].flatten() == roi_idx
roi_betas = data['betas'][:, inds]  # Shape: (81, 100)
```

## Requirements

```bash
uv add scipy numpy
```
