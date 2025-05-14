"""
Analysis utilities for solubility data exploration.
"""
import pandas as pd
import numpy as np
from scipy import stats

def generate_summary_statistics(df, target_col='solubility', exclude_cols=None):
    """
    Generate summary statistics for numerical columns in DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        target_col (str): Name of the target variable column
        exclude_cols (list): Columns to exclude from analysis
        
    Returns:
        pd.DataFrame: DataFrame containing summary statistics
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Select numerical columns excluding specified ones
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    num_cols = [col for col in num_cols if col not in exclude_cols]
    
    # Calculate summary statistics
    summary = df[num_cols].describe().T
    
    # Add additional statistics
    summary['median'] = df[num_cols].median()
    summary['skew'] = df[num_cols].skew()
    summary['kurtosis'] = df[num_cols].kurtosis()
    
    # Calculate correlation with target if it exists
    if target_col in df.columns:
        summary['corr_with_target'] = [df[col].corr(df[target_col]) for col in num_cols]
    
    return summary

def detect_outliers(df, cols=None, method='iqr'):
    """
    Detect outliers in numerical columns of DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        cols (list): Specific columns to check for outliers
        method (str): Method to use ('iqr' or 'zscore')
        
    Returns:
        dict: Dictionary with column names as keys and outlier indices as values
    """
    if cols is None:
        cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    outliers = {}
    
    for col in cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_idx = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_idx = df.index[df[col].notnull()][z_scores > 3]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        outliers[col] = outlier_idx
    
    return outliers

def analyze_feature_importance(df, target_col='solubility', feature_cols=None):
    """
    Calculate feature importance based on correlation with target.
    
    Args:
        df (pd.DataFrame): DataFrame containing features and target
        target_col (str): Name of target column
        feature_cols (list): List of feature column names
        
    Returns:
        pd.Series: Series with feature names as index and importance as values
    """
    if feature_cols is None:
        # Exclude non-numeric and the target column
        feature_cols = [col for col in df.select_dtypes(include=['float64', 'int64']).columns 
                       if col != target_col]
    
    # Calculate absolute correlation with target
    importance = pd.Series(index=feature_cols)
    for col in feature_cols:
        importance[col] = abs(df[col].corr(df[target_col]))
    
    return importance.sort_values(ascending=False)
