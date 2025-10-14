"""
Load and analyze .mat files (AlexNet and Word2Vec features)
"""
import scipy.io as sio
import numpy as np
from pathlib import Path


def load_mat_file(filepath):
    """Load a .mat file and return the data"""
    mat_data = sio.loadmat(filepath)
    # Remove metadata keys
    data_keys = [key for key in mat_data.keys() if not key.startswith('__')]
    return {key: mat_data[key] for key in data_keys}


def analyze_features():
    """Analyze AlexNet and Word2Vec feature matrices"""

    # Load AlexNet features
    alexnet_data = load_mat_file('alexnet.mat')
    alexnet_features = alexnet_data['alexnet']

    # Load Word2Vec features
    word2vec_data = load_mat_file('word2vec.mat')
    word2vec_features = word2vec_data['word2vec']

    print("Feature Matrices")
    print("=" * 50)
    print(f"AlexNet:  {alexnet_features.shape} (stimuli × features)")
    print(f"Word2Vec: {word2vec_features.shape} (stimuli × features)")

    return alexnet_features, word2vec_features


if __name__ == "__main__":
    alexnet, word2vec = analyze_features()
