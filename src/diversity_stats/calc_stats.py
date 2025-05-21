from typing import Optional
from skbio.diversity import alpha_diversity, beta_diversity
from skbio.stats.ordination import pcoa, OrdinationResults
from skbio import DistanceMatrix as DistMat

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
import umap #type: ignore


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
    plt.show()    

    return temp_dataframe

from typing import Union

def make_species_accumulation(sample_names: np.array,
                              species_richness: np.array,
                              grid: Union[str, list] = None,
                              label_y: bool = False,
                              label_x: bool = False,
                              marker: str = 'o',
                              linestyle: str = '-',
                              ) -> plt.figure:
    """
    Make species accumulation curve. It sorts the data in ascending order.
    Arguments:
        sample_names (np.Series): names of samples
        species_richness (np.Series): corresponding species richness
    Returns:
        list: A list of Line2D objects representing the plotted data.
    """
    fig, ax = plt.subplots()
    ax.plot(sample_names, species_richness, marker, linestyle)
    ax.set_xticklabels(sample_names, rotation=90)
    
    if label_y: ax.set_ylabel('Species Richness')
    if label_x: ax.set_xlabel('Samples')
    if grid is not None: ax.grid(True, axis=grid)
    
    return fig

def calc_permanova() -> None:
    
    return


def make_UMAP(
    abundance_matrix: pd.DataFrame,
    n_neighbors: int,
    min_dist: float,
    metadata: pd.DataFrame,
    n_components: int,
    random_state: int,
    metric: str,
    return_reducer: bool = False,
):
    """
    Create a UMAP projection of microbiome data, optionally colored by metadata.
    Parameters:
        abundance_matrix (pd.DataFrame): Samples × features matrix.
        metadata (pd.DataFrame, optional): Sample metadata with sample IDs as index.
        n_neighbors (int): UMAP n_neighbors parameter.
        min_dist (float): UMAP min_dist parameter.
        metric (str): Distance metric for UMAP if abundance_matrix is used.
        n_components (int): Number of UMAP dimensions.
        random_state (int): Random seed for reproducibility.
        return_reducer (bool): If True, also return the UMAP reducer object.
    Returns:
        pd.DataFrame or Tuple[pd.DataFrame, umap.UMAP]: UMAP coordinates with metadata (if provided),
        and optionally the UMAP reducer object.
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=n_components,
        random_state=random_state
    )
    embedding = reducer.fit_transform(abundance_matrix)
    index = abundance_matrix.index

    umap_df = pd.DataFrame(embedding, columns=[f"UMAP{i+1}" for i in range(embedding.shape[1])], index=index)

    if metadata is not None:
        umap_df = umap_df.join(metadata, how="left")

    if return_reducer:
        return umap_df, reducer
    return umap_df


def make_tSNE():
    
    return

