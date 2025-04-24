import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_genus_barplot(genus_df, output_file=None, top_n=15, 
                         width=14, height=8, sort_samples=True,
                         color_palette='tab20'):
    """
    Create a stacked bar chart showing genus-level abundances across all samples
    
    Parameters:
        genus_df (pd.DataFrame): DataFrame with genera as rows and samples as columns
        output_file (str): Path to save the figure (optional)
        top_n (int): Number of top genera to show individually, rest grouped as 'Other'
        width, height (int): Figure dimensions
        sort_samples (bool): Whether to sort samples by total abundance
        color_palette (str): Matplotlib/seaborn color palette name
    """
    # Make a copy to avoid modifying the original
    data = genus_df.copy()
    
    # Calculate total abundance for each genus
    genus_totals = data.sum(axis=1).sort_values(ascending=False)
    
    # Get the top N genera
    top_genera = genus_totals.head(top_n).index.tolist()
    
    # Prepare data for plotting - keep top genera and group others
    plot_data = data.loc[top_genera].copy()
    
    # Add 'Other' category for remaining genera
    other_genera = [g for g in data.index if g not in top_genera]
    if other_genera:
        plot_data.loc['Other'] = data.loc[other_genera].sum()
    
    # Normalize to show relative abundance (convert to percentages)
    plot_data = plot_data / plot_data.sum() * 100
    
    # Sort samples by total abundance if requested
    if sort_samples:
        sample_totals = data.sum()
        sorted_samples = sample_totals.sort_values(ascending=False).index
        plot_data = plot_data[sorted_samples]
    
    # Transpose for stacked bar plotting (samples as index, genera as columns)
    plot_data = plot_data.T
    
    # Create color palette
    colors = sns.color_palette(color_palette, n_colors=len(plot_data.columns))
    
    # Create the plot
    plt.figure(figsize=(width, height))
    
    # Create the stacked bar chart
    plot_data.plot(kind='bar', stacked=True, color=colors, width=0.8)
    
    # Customize the plot
    plt.title('Genus-Level Composition Across Soil Samples', fontsize=16)
    plt.xlabel('Sample', fontsize=12)
    plt.ylabel('Relative Abundance (%)', fontsize=12)
    plt.xticks(rotation=90)
    plt.legend(title='Genus', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save or display
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_file}")
    else:
        plt.show()
    
    return plt.gcf()

# Example usage:
# Assuming your dataframe is called 'genus_abundance_df'
# genus_abundance_df = pd.read_csv("genus_abundance.csv")
# create_genus_barplot(genus_abundance_df, "genus_composition.png")