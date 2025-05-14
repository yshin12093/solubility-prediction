#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration parameters for molecular solubility prediction
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = Path(os.path.join(BASE_DIR.parent.parent, "Project_Data"))
MODEL_DIR = Path(os.path.join(BASE_DIR, "models"))
RESULTS_DIR = Path(os.path.join(BASE_DIR, "results"))

# Create directories if they don't exist
for dir_path in [MODEL_DIR, RESULTS_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Data preprocessing parameters
MAX_SEQ_LENGTH = 128  # Maximum SMILES sequence length
VOCAB_SIZE = 150  # Size of SMILES vocabulary
NUM_AUGMENTATIONS = 3  # Number of non-canonical SMILES augmentations
DESCRIPTOR_NORM_SIGMA_CAP = 5  # Cap molecular descriptors at 5 sigma

# Model parameters
MODEL_TYPE = "encoder-only"  # Transformer model type: encoder-only or encoder-decoder
NUM_LAYERS = 6  # Number of transformer layers
NUM_HEADS = 8  # Number of attention heads
EMBEDDING_DIM = 256  # Embedding dimension
FEEDFORWARD_DIM = 1024  # Feed-forward dimension
DROPOUT_ATTN = 0.1  # Dropout rate in attention layers
DROPOUT_FF = 0.2  # Dropout rate in feed-forward layers

# Training parameters
BATCH_SIZE = 32  # Batch size
NUM_EPOCHS = 30  # Maximum number of epochs
LEARNING_RATE = 5e-5  # Learning rate
WEIGHT_DECAY = 0.01  # Weight decay for regularization
WARMUP_RATIO = 0.1  # Portion of steps for learning rate warmup
PATIENCE = 5  # Early stopping patience
SEED = 42  # Random seed

# Dataset split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Molecular descriptors to calculate
MOLECULAR_DESCRIPTORS = [
    "MolWt",          # Molecular weight
    "LogP",           # LogP (octanol-water partition)
    "NumHDonors",     # Number of hydrogen bond donors
    "NumHAcceptors",  # Number of hydrogen bond acceptors
    "TPSA",           # Topological polar surface area
    "NumAromaticRings",  # Number of aromatic rings
    "NumRotatableBonds",  # Number of rotatable bonds
    "NumHeteroatoms", # Number of heteroatoms
    "FractionCSP3",   # Fraction of carbon atoms that are sp3 hybridized
    "NumRings"        # Total number of rings
]

# Target metric goals
TARGET_RMSE = 1.0  # Target RMSE in log units
TARGET_R2 = 0.7    # Target R²
TARGET_MAE = 0.7   # Target MAE in log units
