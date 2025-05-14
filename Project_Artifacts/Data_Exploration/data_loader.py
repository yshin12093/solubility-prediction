"""
Data loading and preprocessing utilities for solubility prediction.
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import numpy as np

def load_solubility_data(file_path):
    """
    Load solubility data from CSV file and return a DataFrame.
    
    Args:
        file_path (str): Path to the solubility CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing SMILES and solubility data
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded solubility data with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading solubility data: {e}")
        return None

def create_mol_objects(df, smiles_col='smiles'):
    """
    Create RDKit molecule objects from SMILES strings.
    
    Args:
        df (pd.DataFrame): DataFrame containing SMILES strings
        smiles_col (str): Name of column containing SMILES strings
        
    Returns:
        pd.DataFrame: DataFrame with additional 'Molecule' column
    """
    df = df.copy()
    df['Molecule'] = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x))
    # Remove any rows where molecule creation failed
    valid_mols = df['Molecule'].notnull()
    if not all(valid_mols):
        print(f"Warning: {(~valid_mols).sum()} SMILES strings could not be converted to molecules")
        df = df[valid_mols].reset_index(drop=True)
    return df

def calculate_molecular_descriptors(df, feature_list):
    """
    Calculate molecular descriptors for molecules in DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing RDKit molecule objects in 'Molecule' column
        feature_list (list): List of molecular descriptor names to calculate
        
    Returns:
        pd.DataFrame: DataFrame with additional columns for molecular descriptors
    """
    df = df.copy()
    
    # Dictionary mapping feature names to descriptor functions
    descriptor_functions = {
        'MolWt': Descriptors.MolWt,
        'LogP': Descriptors.MolLogP,
        'NumHDonors': Lipinski.NumHDonors,
        'NumHAcceptors': Lipinski.NumHAcceptors,
        'TPSA': Descriptors.TPSA,
        'NumRotatableBonds': Descriptors.NumRotatableBonds,
        'NumAromaticRings': Descriptors.NumAromaticRings,
        'NumHeteroatoms': Descriptors.NumHeteroatoms,
        'NumHeavyAtoms': Descriptors.HeavyAtomCount,
        'FractionCSP3': Descriptors.FractionCSP3
    }
    
    # Calculate each descriptor
    for feature in feature_list:
        if feature in descriptor_functions:
            df[feature] = df['Molecule'].apply(lambda m: descriptor_functions[feature](m) if m else np.nan)
    
    return df
