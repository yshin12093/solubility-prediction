"""
Visualization utilities for solubility data exploration.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from rdkit.Chem import Draw
import io
import base64
from config import FIGURE_SIZE, HIST_BINS, CORRELATION_CMAP

def plot_distribution(df, column, bins=HIST_BINS, save_path=None):
    """
    Create and save a distribution plot for a specific column.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        column (str): Column name to plot
        bins (int): Number of histogram bins
        save_path (str): Path to save the plot (if None, display only)
        
    Returns:
        plt.Figure: Figure object for the plot
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sns.histplot(df[column].dropna(), bins=bins, kde=True, ax=ax)
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_correlation_matrix(df, columns=None, save_path=None):
    """
    Create and save a correlation matrix heatmap.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        columns (list): List of columns to include in correlation matrix
        save_path (str): Path to save the plot (if None, display only)
        
    Returns:
        plt.Figure: Figure object for the plot
    """
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    corr_matrix = df[columns].corr()
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sns.heatmap(corr_matrix, annot=True, cmap=CORRELATION_CMAP, fmt=".2f", ax=ax)
    ax.set_title('Correlation Matrix')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_scatter_with_regression(df, x_col, y_col, save_path=None):
    """
    Create a scatter plot with regression line.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        x_col (str): Column name for x-axis
        y_col (str): Column name for y-axis
        save_path (str): Path to save the plot (if None, display only)
        
    Returns:
        plt.Figure: Figure object for the plot
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sns.regplot(x=x_col, y=y_col, data=df, scatter_kws={'alpha':0.5}, ax=ax)
    ax.set_title(f'{x_col} vs {y_col}')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_molecule_examples(df, smiles_col='smiles', n_molecules=5, save_path=None):
    """
    Create a grid of molecule visualizations.
    
    Args:
        df (pd.DataFrame): DataFrame containing SMILES and molecules
        smiles_col (str): Column containing SMILES strings
        n_molecules (int): Number of molecules to display
        save_path (str): Path to save the plot (if None, return image)
        
    Returns:
        str: Base64 encoded image if save_path is None, otherwise None
    """
    # Get a sample of molecules
    mol_sample = df.sample(min(n_molecules, len(df)))
    mols = [mol for mol in mol_sample['Molecule'] if mol is not None]
    
    if not mols:
        return None
    
    # Generate mol image with legends showing solubility values
    legends = [f"Sol: {sol:.2f}" for sol in mol_sample['solubility']]
    img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(200, 200), legends=legends)
    
    if save_path:
        img.save(save_path)
    else:
        # Convert to base64 for HTML embedding
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def plot_feature_importance(importance_series, save_path=None):
    """
    Create a bar plot of feature importance.
    
    Args:
        importance_series (pd.Series): Series with feature names as index and importance as values
        save_path (str): Path to save the plot (if None, display only)
        
    Returns:
        plt.Figure: Figure object for the plot
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    importance_series.sort_values().plot(kind='barh', ax=ax)
    ax.set_title('Feature Importance (correlation with solubility)')
    ax.set_xlabel('Absolute Correlation')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig
