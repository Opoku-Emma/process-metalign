import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def make_stacked_plot(
    taxonomic_level: pd.DataFrame,
    top_n: int,
    color_palette: str,
    sort_samples = True,
    show_title: bool = True,
    width=14, height=8
    
) -> plt.figure:
    """
    Make a stacked bar chart for a specified taxonomic level
    Arguments:
        taxonomic_level (pd.DataFrame): taxonomic level to be plotted.
        top_n (int): number of names that have to included in the stack
    Returns:
        plt.figure: plotting figure
    """
    
    df = taxonomic_level
    
    sample_totals = taxonomic_level.sum(axis=1).sort_values(ascending=False)
    
    # Top N individuals
    top_individuals = sample_totals.head(top_n).index.to_list()
    
    # Subset data for top individuals
    plotting_data = df.loc[top_individuals]
    
    # Add 'Others' category for the remainder
    others = [name for name in df.index if name not in top_individuals]
    if others:
        plotting_data.loc['Others'] = df.loc[others].sum()
        
    # convert their abundance to percentages
    plotting_data = plotting_data / plotting_data.sum() * 100
    
    # Sort samples by total abundance if requested
    if sort_samples:
        sample_totals = df.sum()
        sorted_samples = sample_totals.sort_values(ascending=False).index
        plotting_data = plotting_data[sorted_samples]
    
    plot_data = plotting_data.transpose()
    
    colors = sns.color_palette(color_palette, len(plot_data))
    plt.figure(figsize=(width, height))
    plot_data.plot(kind='bar', stacked=True, width=0.8, color=colors)
    
    title = ''
    if show_title:
        title = 'Taxon-level'
        
    plt.title(title)
    plt.xlabel('Sample', fontsize=12)
    plt.ylabel('Relative Abundance (%)', fontsize=12)
    plt.xticks(rotation=90)
    plt.legend(title='Genus', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return 


def all_samples_barchart(data: pd.DataFrame, 
                         color_palette,
                         level: str,
                         top_n = 15) -> plt.figure:
    """Create a vertical stacked bar chart."""
    samples_totals = data.sum(axis=1).sort_values(ascending=False)
    samples_totals = samples_totals / samples_totals.sum() * 100
    plot_data = samples_totals.head(top_n)  # Focus on top 10
    plot_data.loc["Others"] = 100 - plot_data.sum()  # Lump remaining into 'Others'
    colors = sns.color_palette(color_palette, len(plot_data))
    fig, ax = plt.subplots()
    
    # Explicitly handle stacking
    bottom = 0
    for i, (genus, value) in enumerate(plot_data.items()):
        ax.bar(["All Samples"], [value], bottom=bottom, color=colors[i], label=genus)
        bottom += value
    
    # Labels & customization
    ax.set_title(f'{level.capitalize()}-Level Composition', fontsize=14)
    ax.set_ylabel('Relative Abundance (%)')
    ax.set_ylim(0, 100)
    ax.set_xticks([])  # Hide x-ticks
    ax.legend(title=f"{level.capitalize()}", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    return fig

def make_barplot(data: pd.DataFrame, sample_id: str, level: str) -> None:
    """
    Makes a barplot of phyla and their relative abundances
    found in a specific sample
    Arguments:
        data (pd.DataFrame): dataframe object containing data for the chosen sample_id./
        It is truncated to show only top 10 phyla
        sample_id (str): The sample_id selected. This also shows as the title of the plot
    Returns:
        None
    """
    sns.barplot(data=data, x=f"{level}_name", y="relative_abundance", hue=f"{level}_name")
    plt.xlabel(f"{level.capitalize()} Name")
    plt.ylabel("Relative abundance (%)")
    plt.title(label=f"{level} Relative Abundance for sample: {sample_id}")
    plt.xticks(rotation=90)
    plt.show()
    
    return

def rank_abundance_plot(
    data: pd.DataFrame,
    sample_id: str,
    level: str = 'species') -> plt.figure:
    """
    Make a rank abundance plot for a specific sample
    Arguments:
        data (pd.DataFrame): Dataframe containing the data for the chosen sample_id.
        sample_id (str): The sample_id selected. This also shows as the title of the plot.
        level (str): Taxonomic level to be plotted. Default is 'species'.
    Returns:
        plt.figure: Plotting figure.
    """
    
    # Melt data
    data_melted = data.melt(var_name=level, value_name='relative_abundance')
    data_melted.set_index(level, inplace=True)
    data_melted.sort(values='relative_abundance', ascending=False, inplace=True)
    ax, fig = plt.subplots(1, 1, figsize=(14, 8))
    ax.plot(data_melted.index, data_melted['relative_abundance'])    
    
    return 