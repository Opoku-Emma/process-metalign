import os
import pathlib
import logging

import pandas as pd
import numpy as np
import sqlite3

from src.diversity_stats import calc_stats
from src import sql_tables
from src import make_rows
from src.assign_lineage_codes import assign_code
from src.my_decorators import my_timer
from src.diversity_stats import make_plot

# This is required for sqlite to properly interpret numpy integers
sqlite3.register_adapter(np.int64, lambda x: int(x))
sqlite3.register_adapter(np.int32, lambda x: int(x))

# Set up configuration for logging errors and other information during runtime
logging.basicConfig(
    filename=f'log/metalign.log',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)
logger = logging.getLogger(__name__)


class MetalignDB:
    """General Class for processing metalign data file"""

    def __init__(self, file_path: str) -> None:
        """
        Arguments:
            path_to_file (str): provide an absolute path to
                the abundance data file obtained from metalign
        """
        # Ensure compatibility across multiple OS
        self.file_path = pathlib.Path(file_path)
        self.database_path = pathlib.Path("data/")

        # Make a directory to store database
        if not os.path.exists(self.database_path):
            os.mkdir(self.database_path) # create the directory if it doesn't exist

        # Get basename of file and remove file extension
        file_basename = os.path.basename(self.file_path)[:-4]

        # Create connection to database. It will be created if it doesn't exist
        self._sql_file_path = os.path.join(self.database_path, file_basename+'.sqlite')
        self._connect()
        self._close()

        # Initialize abundance matrix attribute. This ensures the abundance matrix
        # is created once and use multiple times to save time
        self._abundance_matrix = None

        logger.info("\nChecking if file exists.")
        if self.file_path.exists():
            logger.info("Metalign txt file exists")

            # Check if columns in the file match what we want
            with open(self.file_path) as f:
                first_line = f.readline()
            self._check_column_match(first_line)

            # If the file is compatible, then we create the tables
            if not self._incompatible_format:
                try:
                    logger.info('Reading file')
                    self._read_data()
                    logger.info('Creating SQL tables')
                    self._create_all_taxa_tables()
                except sqlite3.OperationalError as e:
                    logger.error(type(e))
                    logger.info('Looks like the tables have already been created!')
                finally:
                    logger.info('Filling tables')
                    self._fill_tables()
                    self._create_views()
            else:
                raise Exception(f'Exiting. Check log file')
        else:
            raise FileNotFoundError (f"The file path you provided does not exist: \n{self.file_path}")

        # class level hidden attributes. Not intended for the user
        self._GET_DATA = {
        "phylum": self.get_all_phyla,
        "class": self.get_all_class,
        "order": self.get_all_order,
        "family": self.get_all_family,
        "genus": self.get_all_genus,
        "species": self.get_all_species,
    }
        return

    def _check_column_match(self, first_line: str) -> None:
        """
        Checks if the columns in the provided file match the number of columns
        in the template used for this project.
        Arguments:
            first_line (str): The first line of the metalign file.
        Returns:
            None
        """
        logger.info("Checking number of columns.")
        self._incompatible_format = False

        # parse the first line into columns. We assume that the file is tab delimited
        count_columns = len(first_line.split())
        if count_columns != 6:
            self._incompatible_format = True
            logger.error(f"File has unmatching number of columns. It has to contain 6 columns but {count_columns} provided")
        else:
            logger.info("Columns match. Continuing...")
        return

    def run_action(self, 
                   sql: str, 
                   params = None,
                   keep_open: bool = False,
                   commit: bool = False,
                   exec_many: bool = False) -> None:
        """
        Evaluates an SQL string and executes the specified SQL action.
        Arguments:
            sql (str): The SQL query string to be evaluated.
            params: Optional arguments to be passed as parameters to the SQL query.
            keep_open (bool): Indicates whether the database connection should remain open.
            commit (bool): Determines if the SQL action should be committed to the database.
            exec_manu (bool): Defaults to `False`. When `True`, the `executemany` SQL method
                is used to write multiple rows at once.
        Returns:
            None
        """

        self._connect()
        try:
            if exec_many:
                self._conn.executemany(sql, params)
            elif params != None:
                self._conn.execute(sql, params)
            else:
                self._conn.execute(sql)
        except (sqlite3.IntegrityError, sqlite3.OperationalError) as e:
            logger.error(f"Insert Error: {e}")

        if commit: self._conn.commit()
        if not keep_open: self._close() 

        return

    def _create_table(self, sql: str) -> None:
        """
        This helper function just creates the table based on the input sql string.
        
        Arguments:
            sql (str): sql create table string
        Returns:
            None
        """
        self.run_action(sql, commit=True, keep_open=True)
        return

    def _create_all_taxa_tables(self) -> None:
        """
        This creates all required tables
        Returns:
            None
        """
        # Generate SQL create table statements
        _phylum = sql_tables.create_tPhylum()
        _class = sql_tables.create_tClass()
        _order = sql_tables.create_tOrder()
        _family = sql_tables.create_tFamily()
        _genus = sql_tables.create_tGenus()
        _species = sql_tables.create_tSpecies()
        _species_abundance = sql_tables.create_species_abundance()

        table_strings = [_phylum, _class, _order, _family, _genus, _species, _species_abundance]
        for table in table_strings:
            try:
                self._create_table(table)
            except sqlite3.OperationalError as e:
                logger.error(f"Error: {e}")
        self._close()        
        return

    @my_timer
    def _fill_tables(self) -> None:
        """
        This goes through each record in the metalign data file and
        extracts the lineage_id, lineage_name and species_abundance.
        We use a sql.executemany() method to write all records to the database.
        Returns:
            None
        """

        # We only add records to database if this equals the total number of
        # rows in the dataset
        row_count = 0

        # Store data to be inserted into sql database
        species_rows_to_insert = []
        species_abundance = []
        # We store these as sets to remove duplicates
        genus_rows_to_insert = set()
        family_rows_to_insert = set()
        order_rows_to_insert = set()
        class_rows_to_insert = set()
        phylum_rows_to_insert = set()

        total_rows = self._lineages.shape[0] # Get number of rows in the dataset

        for row in self._lineages.to_dict(orient='records'):
            # Subset columns data from individual row
            lineage_id = row["Lineage"]
            lineage_name = row["Taxon"]
            sample_id = row["Sample_ID"]

            # Check for missing values and assign identifiers to them
            # Refer to assign_code for further explanation on identifier assignment
            lineage_id, lineage_name = assign_code(lineage_id, lineage_name)

            row_info = make_rows.make_row_dict(lineage_id, lineage_name, sample_id)
            # Generate SQL INSERT statements
            species_sql = sql_tables.fill_species()
            genus_sql = sql_tables.fill_genus() 
            family_sql = sql_tables.fill_family()
            order_sql = sql_tables.fill_order()
            class_sql = sql_tables.fill_class()
            phylum_sql = sql_tables.fill_phylum()
            species_abundance_sql = sql_tables.species_abundance_sql()

            species_rows_to_insert.append(row_info['species'])
            # Get data for the species abundance in each sample
            species_and_sample_id = row_info["species"][:2]
            species_and_sample_id.append(row["Relative_abundance"])
            species_abundance.append(species_and_sample_id)

            # These check to remove duplicates
            genus_rows_to_insert.add(tuple(row_info['genus']))
            family_rows_to_insert.add(tuple(row_info['family']))
            order_rows_to_insert.add(tuple(row_info['order']))
            class_rows_to_insert.add(tuple(row_info['class']))
            phylum_rows_to_insert.add(tuple(row_info['phylum']))

            row_count += 1
            if row_count >= total_rows:
                # Only add data when total row number is reached
                self.run_action(species_abundance_sql, params=species_abundance, keep_open=True, commit=True, exec_many=True)
                self.run_action(sql=species_sql, params=species_rows_to_insert, keep_open=True, commit=True, exec_many=True)
                self.run_action(sql=genus_sql, params=genus_rows_to_insert, keep_open=True, commit=True, exec_many=True)
                self.run_action(sql=family_sql, params=family_rows_to_insert, keep_open=True, commit=True, exec_many=True)
                self.run_action(sql=order_sql, params=order_rows_to_insert, keep_open=True, commit=True, exec_many=True)
                self.run_action(sql=class_sql, params=class_rows_to_insert, keep_open=True, commit=True, exec_many=True)
                self.run_action(sql=phylum_sql, params=phylum_rows_to_insert, keep_open=True, commit=True, exec_many=True)
        return

    def _create_views(self) -> None:
        """
        Create views for all taxonomic levels
        Returns:
            None
        """
        self._connect()
        self.run_action(sql_tables.vPhylum, keep_open=True, commit=True)
        self.run_action(sql_tables.vClass, keep_open=True, commit=True)
        self.run_action(sql_tables.vOrder, keep_open=True, commit=True)
        self.run_action(sql_tables.vFamily, keep_open=True, commit=True)
        self.run_action(sql_tables.vGenus, commit=True)

        return

    def _read_data(self) -> pd.DataFrame:
        """
        This reads in the metalign data, subsets for only species rows 
        and returns a dataframe with only columns for `Lineage, Taxon, 
        Sample_ID, and Relative abundance`
        Returns:
            pd.Dataframe: contains species lineage, sample_id and abundance
        """
        column_names = [
            "TaxID",
            "Rank",
            "Lineage",
            "Taxon",
            "Relative_abundance",
            "Sample_ID",
        ]

        df = pd.read_csv(
            self.file_path,
            names=column_names,  # Turn this off if the data has headers
            sep="\t",
        )
        # Get the sample names
        self._all_samples = [make_rows.get_sample_name(id) for id in df["Sample_ID"].unique()]

        # Pull out all species.
        # This extracts data for only species level identification
        self._lineages = df[df["Rank"] == "species"]
        self._lineages = self._lineages[["Lineage", "Taxon", "Sample_ID", "Relative_abundance"]]
        return self._lineages

    def get_all_sample_names(self) -> list:
        """
        Prints a list of all sample names
        Returns:
            list: of all sample names
        """
        return self._all_samples
    def get_all_phyla(self, sample_id: str = None) -> pd.DataFrame:
        """
        Retrieves the dataset containing the relative abundance of all phyla
        Arguments:
            sample_id (str): The sample id to be queried. If provided, only phyla
                found in that sample will be returned. Else all phyla in all
                samples will be returned
        Returns:
            pd.DataFrame: Dataframe with relative abundance of phyla present
        """

        sql = sql_tables.get_all_phyla(sample_id)
        self._connect()
        all_phyla = pd.read_sql(sql, self._conn)
        return all_phyla

    def get_all_class(self, sample_id: str = None) -> pd.DataFrame:
        """
        Retrieves the dataset containing the relative abundance of all classes
        Arguments:
            sample_id (str): The sample id to be queried. If provided, only class
                found in that sample will be returned. Else all classes in all
                samples will be returned
        Returns:
            pd.DataFrame: Dataframe with relative abundance of classes present
        """
        sql = sql_tables.get_all_class(sample_id)
        self._connect()
        all_classes = pd.read_sql(sql, self._conn)
        return all_classes

    def get_all_order(self, sample_id: str = None) -> pd.DataFrame:
        """
        Retrieves the dataset containing the relative abundance of all orders
        Arguments:
            sample_id (str): The sample id to be queried. If provided, only orders
                found in that sample will be returned. Else all orders in all
                samples will be returned
        Returns:
            pd.DataFrame: Dataframe with relative abundance of orders present
        """
        sql = sql_tables.get_all_order(sample_id)
        self._connect()
        all_orders = pd.read_sql(sql, self._conn)
        return all_orders

    def get_all_family(self, sample_id: str = None) -> pd.DataFrame:
        """
        Retrieves the dataset containing the relative abundance of all families
        Arguments:
            sample_id (str): The sample id to be queried. If provided, only family
                found in that sample will be returned. Else all family in all
                samples will be returned
        Returns:
            pd.DataFrame: Dataframe with relative abundance of family present
        """

        sql = sql_tables.get_all_family(sample_id)
        self._connect()
        all_family = pd.read_sql(sql, self._conn)
        return all_family

    def get_all_genus(self, sample_id: str = None) -> pd.DataFrame:
        """
        Retrieves the dataset containing the relative abundance of all genera
        Arguments:
            sample_id (str): The sample id to be queried. If provided, only genera
                found in that sample will be returned. Else all genera in all
                samples will be returned
        Returns:
            pd.DataFrame: Dataframe with relative abundance of genera present
        """
        sql = sql_tables.get_all_genus(sample_id)
        self._connect()
        all_genus = pd.read_sql(sql, self._conn)
        return all_genus

    def get_all_species(self, 
                        name: str | None = None, 
                        sample_id: str | None = None
                        ) -> pd.DataFrame:
        """
        Retrieves the dataset containing the relative abundance of all species
        Arguments:
            sample_id (str): The sample id to be queried. If provided, only species
                found in that sample will be returned. Else all species in all
                samples will be returned
            name (str): if provided, the result only shows species 
                with name similar to that,
        Returns:
            pd.DataFrame: with relative abundance of species present
        """

        sql = sql_tables.get_all_species(name, sample_id)
        self._connect()
        all_species = pd.read_sql(sql, self._conn)
        self._close()
        return all_species

    def _make_abundance_matrix(self) -> pd.DataFrame:
        """
        This converts the dataset into an abundance matrix. We will then use it
        as input to calculate diversity metrics later on.
        Arguments:
        
        Returns:
            pd.DataFrame: abundance matrix
        """
        df = self.get_all_species()
        self._abundance_matrix = df.pivot_table(
            index="sample_id",
            columns="species_name",
            values="relative_abundance",
            fill_value=0
        )
        return self._abundance_matrix

    def get_alpha_diversity(self, metric: str = 'shannon') -> pd.Series:
        """
        Calculate the alpha diversity of each sample.
        Arguments:
            metric (str): shannon, simpson, sobs, dominance, chao1, ace, etc.
                refer to `skbio.diversity._driver._get_apha_diversity_metric_map`
                for more metrics 
        Returns:
            pd.Series: list containing diversity indices
        """
        metric = metric.lower()
        if self._abundance_matrix is None:
            self._make_abundance_matrix()
        alph_diversity_metric = calc_stats.calc_alpha_diversity(self._abundance_matrix, metric=metric)
        return alph_diversity_metric

    def get_beta_diversity(self, metric: str = 'braycurtis') -> pd.DataFrame:
        """
        Calculates the beta diversity metric of the entire samples
        Returns:
            DistanceMatrix: dataframe-like object. 
                You can call `beta_diversity_metric.to_data_frame()` 
                to convert it to a `pd.DataFrame` object
        """
        metric = metric.lower()
        if self._abundance_matrix is None:
            self._make_abundance_matrix()
        beta_diversity_metric = calc_stats.calc_beta_diversity(self._abundance_matrix, metric)
        return beta_diversity_metric

    def get_metadata(self, 
                     metadata_path: str,
                     categories: list,
                     index_col: str, 
                     sep='\t'
                     ) -> pd.DataFrame:
        """
        Upload metadata for samples
        Arguments:
            metadata_path (str): path to metadata. 
                First row should be the header row
            categories (list): list of column names in the metadata
                that can be considered for visualization purposes.
            index_col (str): name of column to be used as index. 
                Ideally, it should be the name of the samples.
            sep (str): separator for metadata file. Recommended that data
                is either stored as a csv or tsv file
        Returns:
            pd.DataFrame: dataframe object for metadata
        """
        self._metadata = pd.read_csv(metadata_path, sep=sep, header=0)
        COLUMNS = categories

        # check if category is spelt right and subset that column as well
        for category in categories:
            if category not in self._metadata.columns:
                raise KeyError(f"{category} not found in data. Check the input")
        self._metadata = self._metadata[COLUMNS].set_index(index_col)
        return self._metadata

    def plot_species_accum(
        self,
        show_grids: str | list = None,
        label_y: bool = False,
        label_x: bool = False,
        marker: str = "o",
    ):
        """
        Make a species accumulation curve
        Arguments:
            show_grids (str|list): display the grids on the plot.
                Literal['both', 'x', 'y'] = "both"
            label_y (bool): Add y-axis label to plot
            label_x (bool): Add x-axis label to plot
        Returns:
            plt.figure
        """
        speAbund_sobs = self.get_alpha_diversity("sobs")
        sobs_df = (
            pd.DataFrame(speAbund_sobs, columns=["abundance"])
            .reset_index()
            .sort_values(by="abundance")
        )
        fig = calc_stats.make_species_accumulation(
            sobs_df.sample_id, sobs_df.abundance, show_grids, label_y, label_x, marker
        )

        return fig

    def plot_pcoa(
        self,
        color_by = None,
        method='eigh',
        number_of_dimensions=0,
        ) -> pd.DataFrame:
        """
        Make a PCOA plot
        Returns:
            figure
        """
        distance_matrix = self.get_beta_diversity()
        self._dimensions = calc_stats.make_pcoa_plot(distance_matrix=distance_matrix,
                                  sample_metadata=self._metadata,
                                  color_by=color_by,
                                  method=method,
                                  number_of_dimensions=number_of_dimensions)

        return self._dimensions

    def save_pcoa_dimensions(self, file_path: str, sep: str = ','):
        """
        Save the principal coordinates to a file.
        Arguments:
            file_path (str): name to save file to
        Returns:
            None
        """
        self._dimensions.to_csv(file_path, index=True, sep=sep, index_label="Well")
        return

    def plot_UMAP(self,
                  dissimilarity_matrix: True = None,
                  color_by: str = None
                  ) -> None:

        dissimilarity_matrix=self.get_beta_diversity().to_data_frame()

        make_UMAP = calc_stats.make_UMAP(dissimilarity_matrix = dissimilarity_matrix,
                                        abundance_matrix = self._abundance_matrix,
                                        metadata=self._metadata,
                                        color_by=color_by)

        return make_UMAP

    def plot_tSNE(
        self,
    ) -> None:

        return

    def delete_data(self, path: str) -> None:
        user_input = input(f"Do you want to delete associated database: {path}")
        user_input = user_input.lower()
        if user_input in ["yes", "y"]:
            os.remove(path=path)
        return

    def barplot_by_sample(self, sample_id, level:str, subset=12):
        """
        Makes a barplot for a taxon in a particular sample
        Arguments:
            sample_id (str): sample name to investigate
            level (str): name of taxonomic level
            subset (int): number of groups to display in bar chart.
        Returns:
            figure
        """
        if level not in self._GET_DATA: raise ValueError (f"Level not supported: {level}")
        _func = self._GET_DATA[level]
        df = pd.DataFrame(_func(sample_id)[:subset])
        fig = make_plot.make_barplot(df,sample_id,level)
        return

    def _preprocess_for_barplot(self, 
                                sample: pd.DataFrame, 
                                level: str,
                                choose_samples: list = None,
                                ) -> pd.DataFrame:
        """
        Extract and pivot data for sample to be visualized
        Arguments:
            sample (pd.DataFrame): abundance dataframe to be used. Should
            level (str): taxonomic level
        Returns:
            pd.DataFrame: 
        """

        data_res = sample[[
            f"{level}_name",
            "sample_id",
            "relative_abundance"
        ]]

        # Pivot data to form that can be used for bar chart plotting
        data_res = data_res.pivot_table(
            values='relative_abundance',
            columns='sample_id',
            index=f'{level}_name',
            fill_value=0
        )

        # If user supplied specific sample names, use those to subset the data
        if choose_samples is not None:
            data_res = data_res.loc[:, choose_samples]

        return data_res

    def taxa_level_barplot(
        self,
        level: str = "phylum",
        color_palette: str = "tab10",
        top_n=15,
        choose_samples: list = None,
    ):
        """
        Make barplot (depending on sample if specified) of a taxonomic level
        Returns:
            plt.figure
        """
        # Check if level is valid. superkingdom not included now
        level = level.lower()
        LEVELS = ['phylum', 'class', 'order', 'family', 'genus', 'species']

        # Check to make sure level provided is correct
        if level not in LEVELS:
            raise ValueError(f'{level} not correct')

        taxon_data_res = self._preprocess_for_barplot(
            self._GET_DATA[level](), level, choose_samples
        )

        make_plot.all_samples_barchart(
            taxon_data_res, color_palette, level=level, top_n=top_n
        )
        return

    def _connect(self) -> None:
        """Open connection to database
        Returns:
            None
        """
        self._conn = sqlite3.connect(self._sql_file_path)
        self._connected = True
        self._curs = self._conn.cursor()
        return

    def _close(self) -> None:
        '''Close connection to database'''
        self._conn.close()
        self._connected = False
        return
