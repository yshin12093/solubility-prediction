"""
HTML report generator for solubility data exploration.
"""
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from config import REPORT_TITLE

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def create_html_report(data_overview, summary_stats, feature_importance, 
                       outlier_info, mol_img_base64, correlation_fig, 
                       distribution_figs, scatter_figs, output_path):
    """
    Create HTML report with all exploration results.
    
    Args:
        data_overview (dict): Basic data information
        summary_stats (pd.DataFrame): Statistical summary of features
        feature_importance (pd.Series): Feature importance by correlation
        outlier_info (dict): Outlier information by column
        mol_img_base64 (str): Base64 encoded molecule grid image
        correlation_fig (plt.Figure): Correlation matrix figure
        distribution_figs (dict): Dictionary mapping column names to distribution figures
        scatter_figs (dict): Dictionary mapping column names to scatter figures
        output_path (str): Path to save the HTML report
        
    Returns:
        None
    """
    # Convert plots to base64
    correlation_img = fig_to_base64(correlation_fig)
    distribution_imgs = {col: fig_to_base64(fig) for col, fig in distribution_figs.items()}
    scatter_imgs = {col: fig_to_base64(fig) for col, fig in scatter_figs.items()}
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{REPORT_TITLE}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .plot-container {{
                margin: 20px 0;
            }}
            .highlight {{
                background-color: #ffeaa7;
                padding: 10px;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{REPORT_TITLE}</h1>
            
            <h2>1. Data Overview</h2>
            <div class="highlight">
                <p>Dataset: Solubility (ESOL) Dataset</p>
                <p>Number of compounds: {data_overview['n_compounds']}</p>
                <p>Number of features calculated: {data_overview['n_features']}</p>
                <p>Solubility range: {data_overview['solubility_min']:.2f} to {data_overview['solubility_max']:.2f} log units</p>
            </div>
            
            <h3>Example Molecules</h3>
            <div class="plot-container">
                <img src="data:image/png;base64,{mol_img_base64}" alt="Example Molecules">
            </div>
            
            <h2>2. Essential Quality Checks</h2>
            
            <h3>Missing Values</h3>
            <p>{data_overview['missing_values']}</p>
            
            <h3>Outliers Detection</h3>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Number of Outliers</th>
                    <th>Percentage</th>
                </tr>
    """
    
    # Add outlier info to the table
    for col, indices in outlier_info.items():
        if len(indices) > 0:
            pct = len(indices) / data_overview['n_compounds'] * 100
            html_content += f"""
                <tr>
                    <td>{col}</td>
                    <td>{len(indices)}</td>
                    <td>{pct:.2f}%</td>
                </tr>
            """
    
    html_content += """
            </table>
            
            <h2>3. Basic Variable Analysis</h2>
            
            <h3>Summary Statistics</h3>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Mean</th>
                    <th>Std</th>
                    <th>Min</th>
                    <th>25%</th>
                    <th>50% (Median)</th>
                    <th>75%</th>
                    <th>Max</th>
                    <th>Correlation with Solubility</th>
                </tr>
    """
    
    # Add summary stats to the table
    for feature, row in summary_stats.iterrows():
        html_content += f"""
                <tr>
                    <td>{feature}</td>
                    <td>{row['mean']:.2f}</td>
                    <td>{row['std']:.2f}</td>
                    <td>{row['min']:.2f}</td>
                    <td>{row['25%']:.2f}</td>
                    <td>{row['median']:.2f}</td>
                    <td>{row['75%']:.2f}</td>
                    <td>{row['max']:.2f}</td>
                    <td>{row.get('corr_with_target', 'N/A'):.2f}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h3>Feature Importance</h3>
            <div class="plot-container">
                <img src="data:image/png;base64,{0}" alt="Feature Importance">
            </div>
            
            <h3>Correlation Matrix</h3>
            <div class="plot-container">
                <img src="data:image/png;base64,{1}" alt="Correlation Matrix">
            </div>
            
            <h3>Distribution Plots</h3>
    """.format(
        fig_to_base64(plt.figure()),  # Placeholder for feature importance
        correlation_img
    )
    
    # Add distribution plots
    for col, img in distribution_imgs.items():
        html_content += f"""
            <div class="plot-container">
                <h4>{col} Distribution</h4>
                <img src="data:image/png;base64,{img}" alt="{col} Distribution">
            </div>
        """
    
    # Add scatter plots
    html_content += """
            <h3>Relationship with Solubility</h3>
    """
    
    for col, img in scatter_imgs.items():
        html_content += f"""
            <div class="plot-container">
                <h4>{col} vs Solubility</h4>
                <img src="data:image/png;base64,{img}" alt="{col} vs Solubility">
            </div>
        """
    
    # Close the HTML
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {output_path}")
