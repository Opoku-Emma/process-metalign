from typing import Optional
from skbio.diversity import alpha_diversity, beta_diversity
from skbio.stats.ordination import pcoa, OrdinationResults
from skbio import DistanceMatrix as DistMat

# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import umap.umap_ as umap

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
    
    # for idx in range(len(temp_dataframe)):
    #     plt.text(
    #         x=temp_dataframe['PC1'].iloc[idx]+0.005,
    #         y=temp_dataframe['PC2'].iloc[idx]+0.005,
    #         s=temp_dataframe.index[idx])
    plt.show()    

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
    sns.set(
        style='white',
        context='paper',
        rc={'figure.figsize': (14, 10)}
    )

    if dissimilarity_matrix is None:
        # Normalize and reduce from abundance
        data = abundance_matrix.values
        scaled_data = StandardScaler().fit_transform(data)
        reducer = umap.UMAP(random_state=1234)
        embedding = reducer.fit_transform(scaled_data)
    else:
        # Use dissimilarity (e.g., Bray-Curtis)
        reducer = umap.UMAP(metric='precomputed', random_state=1234)
        embedding = reducer.fit_transform(dissimilarity_matrix.values)

    # Plot
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=60,
        alpha=0.7,
        edgecolor='k'
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of Microbiome Diversity', fontsize=20)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.tight_layout()
    plt.show()

    return embedding


from typing import Tuple, Optional, Dict, Any, Union


def make_UMAP(
    abundance_matrix: Optional[pd.DataFrame] = None,
    dissimilarity_matrix: Optional[pd.DataFrame] = None,
    metadata: Optional[pd.DataFrame] = None,
    color_by: Optional[str] = None,
    umap_params: Optional[Dict[str, Any]] = None,
    plot_params: Optional[Dict[str, Any]] = None,
    return_reducer: bool = False,
    show_plot: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, umap.UMAP]]:
    """
    Create a UMAP projection of microbiome data, optionally colored by metadata.
    
    Parameters:
        abundance_matrix (pd.DataFrame, optional): Samples × features matrix.
        dissimilarity_matrix (pd.DataFrame, optional): Precomputed dissimilarity matrix (e.g., Bray-Curtis).
        metadata (pd.DataFrame, optional): Sample metadata with sample IDs as index.
        color_by (str, optional): Column in `metadata` to use for coloring points.
        umap_params (dict, optional): Parameters to pass to UMAP constructor.
        plot_params (dict, optional): Parameters for plot customization.
        return_reducer (bool): Whether to return the UMAP reducer object along with results.
        show_plot (bool): Whether to display the plot.
        
    Returns:
        pd.DataFrame or Tuple[pd.DataFrame, umap.UMAP]: UMAP coordinates with metadata (if provided),
        and optionally the UMAP reducer object.
    """
    # Validate input
    if abundance_matrix is None and dissimilarity_matrix is None:
        raise ValueError("Either abundance_matrix or dissimilarity_matrix must be provided.")
    
    # Default parameters
    default_umap_params = {"random_state": 42}
    default_plot_params = {
        "figsize": (8, 6),
        "title": "UMAP Projection of Microbiome Diversity",
        "title_fontsize": 18,
        "point_size": 80,
        "alpha": 0.8,
        "palette": "Set2",
    }
    
    # Update with user parameters if provided
    if umap_params:
        default_umap_params.update(umap_params)
    
    if plot_params:
        default_plot_params.update(plot_params)
    
    # Prepare data and fit UMAP
    if dissimilarity_matrix is not None:
        # Validate dissimilarity matrix is symmetric and square
        if not _is_valid_distance_matrix(dissimilarity_matrix):
            raise ValueError("Dissimilarity matrix must be symmetric and square.")
        
        default_umap_params["metric"] = "precomputed"
        reducer = umap.UMAP(**default_umap_params)
        embedding = reducer.fit_transform(dissimilarity_matrix.values)
        sample_ids = dissimilarity_matrix.index
    else:
        # Use abundance matrix
        data = StandardScaler().fit_transform(abundance_matrix.values)
        reducer = umap.UMAP(**default_umap_params)
        embedding = reducer.fit_transform(data)
        sample_ids = abundance_matrix.index
    
    # Create DataFrame with UMAP results
    umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'], index=sample_ids)
    
    # Add metadata if available
    if metadata is not None and color_by is not None:
        if color_by not in metadata.columns:
            raise ValueError(f"'{color_by}' is not a column in metadata.")
        
        # Check for matching indices between data and metadata
        missing_samples = [idx for idx in umap_df.index if idx not in metadata.index]
        if missing_samples:
            print(f"Warning: {len(missing_samples)} samples in data not found in metadata.")
        
        # Only add metadata for samples that exist in both datasets
        common_samples = umap_df.index.intersection(metadata.index)
        umap_df = umap_df.loc[common_samples]
        umap_df[color_by] = metadata.loc[umap_df.index, color_by]
    
    # Create plot if requested
    if show_plot:
        # Set seaborn theme
        sns.set_theme(style='white', context='paper')
        
        # Create figure
        plt.figure(figsize=default_plot_params["figsize"])
        
        # Plot with or without color
        if color_by and metadata is not None:
            sns.scatterplot(
                data=umap_df,
                x="UMAP1", y="UMAP2",
                hue=color_by,
                palette=default_plot_params["palette"],
                s=default_plot_params["point_size"],
                alpha=default_plot_params["alpha"],
                edgecolor="k"
            )
            plt.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            plt.scatter(
                umap_df["UMAP1"], 
                umap_df["UMAP2"], 
                s=default_plot_params["point_size"], 
                alpha=default_plot_params["alpha"], 
                edgecolor="k"
            )
        
        plt.title(default_plot_params["title"], fontsize=default_plot_params["title_fontsize"])
        plt.tight_layout()
        plt.show()
    
    # Return results
    if return_reducer:
        return umap_df, reducer
    else:
        return umap_df


def _is_valid_distance_matrix(matrix: pd.DataFrame) -> bool:
    """
    Check if a matrix is a valid distance/dissimilarity matrix:
    - Must be square (n×n)
    - Must be symmetric
    - Diagonal should be zeros
    
    Parameters:
        matrix (pd.DataFrame): Matrix to check
    
    Returns:
        bool: True if valid distance matrix, False otherwise
    """
    # Check if square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Check if symmetric (within numerical precision)
    if not np.allclose(matrix, matrix.T):
        return False
    
    # Check if diagonal is all zeros (within tolerance)
    diag = np.diag(matrix)
    if not np.allclose(diag, np.zeros_like(diag)):
        return False
    
    return True


def make_tSNE():
    
    return

