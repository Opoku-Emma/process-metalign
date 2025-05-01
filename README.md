# Microbiome Data Analysis Using Metalign Output and SQL Database Management
The goal of this project is to provide a streamlined and easy analysis of microbiome sequencing data that has been generated using metalign.
Given that you are new to [metalign](https://github.com/nlapier2/Metalign) and how it works, you might want to check out their documentation page [here](https://github.com/nlapier2/Metalign).

## Getting Started
### src/
contains necessary scripts to run analysis\
-- assign_lineage_codes.py\
-- my_decorators.py\
-- preprocess_data.py\
-- sql_tables.py\
-- make_rows.py\
-- metalign_analysis.py

### data/
Contains demo data for analysis.
- leaf_all.nostrain.txt
- leaf_phenotype.txt

### main_analysis.py
This runs a demo on how the script works. It reads in the metalign file and metadata file.
You only need to provide the metalign file path which provides you features like:
- Calculating alpha and beta diversity
- Making plots of taxonomic groups/levels
- Pulling data on specified taxonomic groups/levels
  
If metadata_file is added, further functionality is added to make Principal Coordinate Plots and coloring based on specified feature

Then it runs a simple analysis and returns a Principal Coordinate Analysis plot colored by site.

## Usage
Run notebook [analysis notebook](https://github.com/Opoku-Emma/process-metalign/blob/main/analysis_notebook.ipynb) to see how it works