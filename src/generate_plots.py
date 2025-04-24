import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


