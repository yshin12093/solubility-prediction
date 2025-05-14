#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data processing utilities for molecular solubility prediction
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Union, Optional
import random
import config

class SMILESTokenizer:
    """Tokenizer for SMILES strings"""
    
    def __init__(self, max_length=config.MAX_SEQ_LENGTH):
        """
        Initialize the SMILES tokenizer
        
        Args:
            max_length: Maximum sequence length
        """
        self.max_length = max_length
        self.vocab = None
        self.token_to_idx = None
        self.idx_to_token = None
        self.special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]
        
    def build_vocabulary(self, smiles_list: List[str]) -> None:
        """
        Build vocabulary from a list of SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
        """
        # Collect all unique tokens
        unique_tokens = set()
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Get canonical SMILES
                smiles = Chem.MolToSmiles(mol)
                for token in self._tokenize_smiles(smiles):
                    unique_tokens.add(token)
        
        # Create vocabulary with special tokens first
        self.vocab = self.special_tokens + sorted(list(unique_tokens))
        
        # Create token to index mapping
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        
        print(f"Vocabulary built with {len(self.vocab)} tokens")
    
    def _tokenize_smiles(self, smiles: str) -> List[str]:
        """
        Tokenize SMILES string into individual characters and tokens
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of tokens
        """
        # Character-level tokenization with some special handling for multi-character tokens
        tokens = []
        i = 0
        while i < len(smiles):
            # Handle two-character tokens like 'Cl', 'Br', etc.
            if i < len(smiles) - 1 and smiles[i:i+2] in ['Cl', 'Br', 'Si', 'Se', 'Na', 'Li', 'Mg']:
                tokens.append(smiles[i:i+2])
                i += 2
            else:
                tokens.append(smiles[i])
                i += 1
        return tokens
    
    def encode(self, smiles: str) -> List[int]:
        """
        Encode SMILES string to token IDs
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of token IDs
        """
        if not self.token_to_idx:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")
        
        tokens = self._tokenize_smiles(smiles)
        # Add CLS token at the beginning
        token_ids = [self.token_to_idx["[CLS]"]]
        
        # Convert tokens to IDs
        for token in tokens:
            if token in self.token_to_idx:
                token_ids.append(self.token_to_idx[token])
            else:
                token_ids.append(self.token_to_idx["[UNK]"])
        
        # Add SEP token at the end
        token_ids.append(self.token_to_idx["[SEP]"])
        
        # Pad or truncate
        if len(token_ids) < self.max_length:
            token_ids += [self.token_to_idx["[PAD]"]] * (self.max_length - len(token_ids))
        elif len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length-1] + [self.token_to_idx["[SEP]"]]
            
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to SMILES string
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            SMILES string
        """
        if not self.idx_to_token:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")
        
        # Convert IDs to tokens, skipping special tokens
        tokens = []
        for idx in token_ids:
            token = self.idx_to_token.get(idx)
            if token and token not in self.special_tokens:
                tokens.append(token)
                
        return "".join(tokens)


class MolecularDescriptorCalculator:
    """Calculate and normalize molecular descriptors"""
    
    def __init__(self, descriptors=config.MOLECULAR_DESCRIPTORS):
        """
        Initialize descriptor calculator
        
        Args:
            descriptors: List of descriptors to calculate
        """
        self.descriptors = descriptors
        self.scaler = StandardScaler()
        self.descriptor_functions = {
            "MolWt": Descriptors.MolWt,
            "LogP": Descriptors.MolLogP,
            "NumHDonors": Lipinski.NumHDonors,
            "NumHAcceptors": Lipinski.NumHAcceptors,
            "TPSA": Descriptors.TPSA,
            "NumAromaticRings": lambda m: Lipinski.NumAromaticRings(m),
            "NumRotatableBonds": Descriptors.NumRotatableBonds,
            "NumHeteroatoms": Descriptors.NumHeteroatoms,
            "FractionCSP3": Descriptors.FractionCSP3,
            "NumRings": Descriptors.RingCount
        }
        self.is_fitted = False
        
    def calculate(self, smiles: str) -> Dict[str, float]:
        """
        Calculate descriptors for a single SMILES string
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary of descriptor values
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {desc: np.nan for desc in self.descriptors}
        
        result = {}
        for desc in self.descriptors:
            if desc in self.descriptor_functions:
                try:
                    value = self.descriptor_functions[desc](mol)
                    result[desc] = value
                except:
                    result[desc] = np.nan
            else:
                result[desc] = np.nan
                
        return result
    
    def calculate_batch(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Calculate descriptors for a batch of SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            DataFrame of descriptor values
        """
        results = []
        for smiles in smiles_list:
            results.append(self.calculate(smiles))
            
        return pd.DataFrame(results)
    
    def fit(self, descriptors_df: pd.DataFrame) -> None:
        """
        Fit the scaler to the descriptors
        
        Args:
            descriptors_df: DataFrame of descriptors
        """
        # Replace NaN with median for each column
        df_clean = descriptors_df.fillna(descriptors_df.median())
        self.scaler.fit(df_clean)
        self.is_fitted = True
        
    def transform(self, descriptors_df: pd.DataFrame, sigma_cap=config.DESCRIPTOR_NORM_SIGMA_CAP) -> np.ndarray:
        """
        Transform descriptors using fitted scaler
        
        Args:
            descriptors_df: DataFrame of descriptors
            sigma_cap: Cap values at this many standard deviations
            
        Returns:
            Normalized descriptor array
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit first.")
        
        # Replace NaN with median for each column
        df_clean = descriptors_df.fillna(descriptors_df.median())
        
        # Transform data
        scaled_data = self.scaler.transform(df_clean)
        
        # Cap extreme values
        if sigma_cap:
            scaled_data = np.clip(scaled_data, -sigma_cap, sigma_cap)
            
        return scaled_data


def generate_noncanonical_smiles(smiles: str, num_variants: int = config.NUM_AUGMENTATIONS) -> List[str]:
    """
    Generate non-canonical SMILES variants for data augmentation
    
    Args:
        smiles: Original SMILES string
        num_variants: Number of variants to generate
        
    Returns:
        List of SMILES variants including the original
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [smiles]
    
    variants = [smiles]  # Include the original canonical SMILES
    
    # Generate additional random SMILES variants
    for _ in range(num_variants - 1):
        if mol.GetNumAtoms() > 1:
            # Random atom ordering using a random seed
            random_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False, isomericSmiles=True)
            if random_smiles not in variants:
                variants.append(random_smiles)
        
    # Fill with duplicates if we couldn't generate enough variants
    while len(variants) < num_variants:
        variants.append(smiles)
        
    return variants


class SolubilityDataset(Dataset):
    """PyTorch dataset for molecular solubility data"""
    
    def __init__(
        self, 
        smiles_list: List[str], 
        solubility_values: List[float],
        tokenizer: SMILESTokenizer,
        descriptor_calculator: MolecularDescriptorCalculator,
        augment: bool = False,
        num_augmentations: int = 1
    ):
        """
        Initialize the dataset
        
        Args:
            smiles_list: List of SMILES strings
            solubility_values: List of solubility values
            tokenizer: SMILES tokenizer
            descriptor_calculator: Molecular descriptor calculator
            augment: Whether to augment data with non-canonical SMILES
            num_augmentations: Number of augmentations per molecule
        """
        self.tokenizer = tokenizer
        self.descriptor_calculator = descriptor_calculator
        self.augment = augment
        self.num_augmentations = num_augmentations
        
        # Store original data
        self.original_smiles = smiles_list
        self.original_solubility = solubility_values
        
        # Prepare augmented data if needed
        if augment and num_augmentations > 1:
            self._augment_data()
        else:
            self.smiles = smiles_list
            self.solubility = solubility_values
            
        # Calculate descriptors for all SMILES
        self.descriptors_df = self.descriptor_calculator.calculate_batch(self.smiles)
        
        # Ensure the descriptor calculator is fitted
        if not self.descriptor_calculator.is_fitted:
            self.descriptor_calculator.fit(self.descriptors_df)
            
        # Transform descriptors
        self.normalized_descriptors = self.descriptor_calculator.transform(self.descriptors_df)
        
    def _augment_data(self) -> None:
        """Augment data with non-canonical SMILES variants"""
        augmented_smiles = []
        augmented_solubility = []
        
        for smiles, solubility in zip(self.original_smiles, self.original_solubility):
            variants = generate_noncanonical_smiles(smiles, self.num_augmentations)
            augmented_smiles.extend(variants)
            augmented_solubility.extend([solubility] * len(variants))
            
        self.smiles = augmented_smiles
        self.solubility = augmented_solubility
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.smiles)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with input_ids, attention_mask, descriptor, and solubility
        """
        # Encode SMILES to token IDs
        input_ids = torch.tensor(self.tokenizer.encode(self.smiles[idx]), dtype=torch.long)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.tokenizer.token_to_idx["[PAD]"]).long()
        
        # Get descriptors
        descriptor = torch.tensor(self.normalized_descriptors[idx], dtype=torch.float)
        
        # Get solubility
        solubility = torch.tensor(self.solubility[idx], dtype=torch.float)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "descriptor": descriptor,
            "solubility": solubility
        }


def prepare_data(
    data_path: str,
    smiles_col: str = "SMILES",
    solubility_col: str = "Solubility",
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    augment_train: bool = True,
    random_state: int = config.SEED
) -> Tuple[SolubilityDataset, SolubilityDataset, SolubilityDataset]:
    """
    Prepare train, validation, and test datasets
    
    Args:
        data_path: Path to data file (CSV)
        smiles_col: Column name for SMILES strings
        solubility_col: Column name for solubility values
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        augment_train: Whether to augment training data
        random_state: Random seed
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Check for required columns
    if smiles_col not in df.columns or solubility_col not in df.columns:
        raise ValueError(f"Data must contain '{smiles_col}' and '{solubility_col}' columns")
    
    # Split data into train, validation, and test sets
    train_val_df, test_df = train_test_split(
        df, test_size=test_ratio, random_state=random_state, stratify=_stratify_by_solubility(df[solubility_col])
    )
    
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_ratio/(train_ratio + val_ratio), 
        random_state=random_state,
        stratify=_stratify_by_solubility(train_val_df[solubility_col])
    )
    
    # Initialize tokenizer and build vocabulary from all SMILES
    tokenizer = SMILESTokenizer()
    tokenizer.build_vocabulary(df[smiles_col].tolist())
    
    # Initialize descriptor calculator and fit on all data
    descriptor_calculator = MolecularDescriptorCalculator()
    all_descriptors_df = descriptor_calculator.calculate_batch(df[smiles_col].tolist())
    descriptor_calculator.fit(all_descriptors_df)
    
    # Create datasets
    train_dataset = SolubilityDataset(
        train_df[smiles_col].tolist(),
        train_df[solubility_col].tolist(),
        tokenizer,
        descriptor_calculator,
        augment=augment_train,
        num_augmentations=config.NUM_AUGMENTATIONS
    )
    
    val_dataset = SolubilityDataset(
        val_df[smiles_col].tolist(),
        val_df[solubility_col].tolist(),
        tokenizer,
        descriptor_calculator,
        augment=False
    )
    
    test_dataset = SolubilityDataset(
        test_df[smiles_col].tolist(),
        test_df[solubility_col].tolist(),
        tokenizer,
        descriptor_calculator,
        augment=False
    )
    
    return train_dataset, val_dataset, test_dataset, tokenizer, descriptor_calculator


def _stratify_by_solubility(solubility_values: pd.Series) -> pd.Series:
    """
    Create stratification labels based on solubility quartiles
    
    Args:
        solubility_values: Series of solubility values
        
    Returns:
        Series of stratification labels
    """
    # Create quartile labels for stratified splitting
    return pd.qcut(solubility_values, 4, labels=False)


def create_dataloaders(
    train_dataset: SolubilityDataset,
    val_dataset: SolubilityDataset,
    test_dataset: SolubilityDataset,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test datasets
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
