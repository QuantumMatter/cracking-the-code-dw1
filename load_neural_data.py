"""
Load and analyze neural ROI data from fMRI

DATA STRUCTURE:
--------------
The pickle file contains a dictionary with 4 keys:

1. 'rois' (1×7 array): Names of 7 brain regions
   ['EVC', 'OPA', 'PPA', 'LOC', 'PFS', 'OFA', 'FFA']

2. 'indices' (1×696 array): Maps each voxel to its ROI (values 1-7)
   Example: indices[0] = 1 means voxel 0 belongs to ROI 1 (EVC)

3. 'betas' (81×696 array): Neural responses
   - Rows = 81 stimuli
   - Columns = 696 voxels
   - betas[i, j] = response of voxel j to stimulus i

4. 'conds' (81×1 array): Stimulus labels
   - conds[i] = name of stimulus i (e.g., 'airplane', 'chair')
   - ALIGNED WITH BETAS: betas[i, :] is the response to conds[i]

USAGE EXAMPLE:
-------------
# Get FFA (ROI 7) responses to "airplane" (stimulus 0):
ffa_mask = indices.flatten() == 7
airplane_ffa_response = betas[0, ffa_mask]  # Shape: (100,)

# Get all stimuli responses for EVC (ROI 1):
evc_mask = indices.flatten() == 1
evc_responses = betas[:, evc_mask]  # Shape: (81, 100)
"""

import pickle
import numpy as np


def load_neural_data(filepath='python_roi_data.pickle'):
    """Load neural ROI data from pickle file"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def get_roi_data(data, roi_name):
    """
    Extract data for a specific ROI

    Args:
        data: Dictionary from load_neural_data()
        roi_name: Name of ROI (e.g., 'FFA', 'EVC')

    Returns:
        roi_betas: (81 × N) array where N = number of voxels in that ROI
    """
    rois = data['rois'].flatten()
    indices = data['indices'].flatten()
    betas = data['betas']

    # Find ROI index
    roi_idx = None
    for i, roi in enumerate(rois, start=1):
        if roi[0] == roi_name:
            roi_idx = i
            break

    if roi_idx is None:
        raise ValueError(f"ROI '{roi_name}' not found. Available: {[r[0] for r in rois]}")

    # Extract voxels belonging to this ROI
    roi_mask = indices == roi_idx
    roi_betas = betas[:, roi_mask]

    return roi_betas


def analyze_neural_data():
    """Load and print summary of neural data"""
    data = load_neural_data()

    rois = data['rois'].flatten()
    indices = data['indices'].flatten()
    betas = data['betas']
    conds = data['conds'].flatten()

    print("Neural ROI Data")
    print("=" * 50)
    print(f"Stimuli: {len(conds)}")
    print(f"Total voxels: {betas.shape[1]}")
    print(f"ROIs: {len(rois)}")
    print()

    print("ROI Breakdown:")
    print("-" * 50)
    for roi_idx, roi in enumerate(rois, start=1):
        voxel_count = np.sum(indices == roi_idx)
        print(f"{roi[0]:4s}: {voxel_count:3d} voxels")

    print()
    print(f"Example conditions: {', '.join([c[0] for c in conds[:5]])}")

    return data


if __name__ == "__main__":
    data = analyze_neural_data()

    # Example: Get FFA responses
    print("\nExample: FFA responses")
    print("-" * 50)
    ffa_data = get_roi_data(data, 'FFA')
    print(f"Shape: {ffa_data.shape} (stimuli × voxels)")
    print(f"Mean response: {np.mean(ffa_data):.4f}")
