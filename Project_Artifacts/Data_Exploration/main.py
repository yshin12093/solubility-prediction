"""
Main script for solubility data exploration.
This script orchestrates the entire data exploration process.
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data_loader import load_solubility_data, create_mol_objects, calculate_molecular_descriptors
from data_analyzer import generate_summary_statistics, detect_outliers, analyze_feature_importance
from data_visualizer import (plot_distribution, plot_correlation_matrix, 
                            plot_scatter_with_regression, plot_molecule_examples,
                            plot_feature_importance)
from report_generator import create_html_report
from config import SOLUBILITY_DATA, MOLECULE_FEATURES, REPORT_PATH, OUTPUT_PATH

def main():
    """Main function to orchestrate the data exploration process."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Load data
    df = load_solubility_data(SOLUBILITY_DATA)
    if df is None:
        print("Error: Failed to load data. Exiting.")
        return
    
    # Create RDKit molecule objects
    df = create_mol_objects(df)
    
    # Calculate molecular descriptors
    df = calculate_molecular_descriptors(df, MOLECULE_FEATURES)
    
    # Data overview
    data_overview = {
        'n_compounds': len(df),
        'n_features': len(MOLECULE_FEATURES),
        'solubility_min': df['solubility'].min(),
        'solubility_max': df['solubility'].max(),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    # Summary statistics
    summary_stats = generate_summary_statistics(df, exclude_cols=['Molecule'])
    
    # Feature importance
    feature_importance = analyze_feature_importance(df, feature_cols=MOLECULE_FEATURES)
    
    # Feature importance plot
    feature_importance_fig = plot_feature_importance(feature_importance, 
                                                    save_path=f"{OUTPUT_PATH}/feature_importance.png")
    
    # Outlier detection
    outlier_info = detect_outliers(df, cols=MOLECULE_FEATURES + ['solubility'])
    
    # Correlation matrix
    correlation_fig = plot_correlation_matrix(df, columns=MOLECULE_FEATURES + ['solubility'], 
                                             save_path=f"{OUTPUT_PATH}/correlation_matrix.png")
    
    # Distribution plots
    distribution_figs = {}
    for feature in MOLECULE_FEATURES + ['solubility']:
        fig = plot_distribution(df, feature, save_path=f"{OUTPUT_PATH}/distribution_{feature}.png")
        distribution_figs[feature] = fig
    
    # Scatter plots with regression
    scatter_figs = {}
    for feature in MOLECULE_FEATURES:
        fig = plot_scatter_with_regression(df, feature, 'solubility', 
                                         save_path=f"{OUTPUT_PATH}/scatter_{feature}.png")
        scatter_figs[feature] = fig
    
    # Molecule examples
    mol_img_base64 = plot_molecule_examples(df, n_molecules=6, 
                                           save_path=f"{OUTPUT_PATH}/molecule_examples.png")
    
    # Generate HTML report
    create_html_report(
        data_overview=data_overview,
        summary_stats=summary_stats,
        feature_importance=feature_importance,
        outlier_info=outlier_info,
        mol_img_base64=mol_img_base64,
        correlation_fig=correlation_fig,
        distribution_figs=distribution_figs,
        scatter_figs=scatter_figs,
        output_path=REPORT_PATH
    )
    
    print("Data exploration completed successfully.")
    print(f"HTML report saved to {REPORT_PATH}")

if __name__ == "__main__":
    main()
