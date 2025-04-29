from skbio.diversity import alpha_diversity, beta_diversity
from skbio.stats.ordination import pcoa, OrdinationResults
from skbio import DistanceMatrix as DistMat

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import umap.umap_ as umap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calc_alpha_diversity(
    data: pd.DataFrame,
    metric: str = "shannon",
) -> pd.Series:
    """
    Calculate alpha diversity for dataframe.
    Arguments:
        data (pd.DataFrame): species abundance data matrix
    Returns:
        pd.Series: results
    """
    # data has to have the sample names as the row index
    diversity = alpha_diversity(metric=metric, counts=data, ids=data.index)
    return diversity


def calc_beta_diversity(
    data: pd.DataFrame,
    metric: str = "braycurtis",
) -> DistMat:
    """
    Documentation
    Arguments:
        data (pd.DataFrame): pandas dataframe object with species abundance data
    Returns:
        np.array: array-like object containing beta diversity matrix
    """
    diversity = beta_diversity(counts=data, metric=metric, ids=data.index)
    return diversity


def make_pcoa_plot(
    distance_matrix: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    color_by: str,
    method: str = "eigh",
    number_of_dimensions=2,
    ) -> OrdinationResults:
    """
    Compute Principal Coordinate Analysis
    Arguments:
        distance_matrix (pd.DataFrame): 
        sample_metadata (pd.DataFrame):
            A pandas DataFrame where each row represents a sample and the index contains sample IDs.
            Used for grouping or coloring samples in the PCoA plot via the specified `color_by` column.
        color_by (str):
        method (str): 
        number_of_dimensions (int):
    Returns:
        figure
    """
    pcoa_results = pcoa(
        distance_matrix=distance_matrix,
        method=method,
        number_of_dimensions=number_of_dimensions,
    )
    
    pc1 = pcoa_results.samples['PC1']
    pc2 = pcoa_results.samples['PC2']
    temp_dataframe = pd.DataFrame({'PC1': pc1, 'PC2': pc2})
    temp_dataframe = temp_dataframe.join(sample_metadata)
    
    sns.scatterplot(data=temp_dataframe, 
                          x='PC1', 
                          y='PC2', 
                          hue=color_by if color_by is not None else None)
    plt.xlabel(f'PC1 ({pcoa_results.proportion_explained.iloc[0]:.2%})')
    plt.ylabel(f'PC2 ({pcoa_results.proportion_explained.iloc[1]:.2%})')

    return temp_dataframe


def make_UMAP1(
    abundance_matrix: pd.DataFrame,
    dissimilarity_matrix: pd.DataFrame = None,
):
    """
    Create a UMAP projection of microbiome data.
    Parameters:
        abundance_matrix (pd.DataFrame): Samples × features matrix (e.g., taxa counts or relative abundance).
        dissimilarity_matrix (pd.DataFrame, optional): Optional precomputed distance matrix (e.g., Bray-Curtis).
    Returns:
        np.ndarray: 2D UMAP embedding coordinates.
    """
    return

def make_UMAP(
    abundance_matrix: pd.DataFrame = None,
    dissimilarity_matrix: pd.DataFrame = None,
    metadata: pd.DataFrame = None) -> None:
    return

def make_tSNE():
    
    return

