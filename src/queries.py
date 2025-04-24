"""
This module contains query functions for interacting with the database
Functions:
    get_some_species(species_name: str) -> str:
        Generates an SQL query to retrieve species information based on a partial match of the species name.
Attributes:
    __author__ (str): Emmanuel Opoku
    __version__ (float): 1.0
This module provides series of query statements
that can be called to  communicate to the database
"""

__author__ = 'Emmanuel Opoku'
__version__ = 1.0

##############################################################################
##----------------------------  CREATE VIEWS  ------------------------------##
##############################################################################

vGenus = """
CREATE VIEW vGenus AS
WITH
species AS
(
    SELECT species_id, sample_id, genus_id
    FROM tSpecies
),

genus AS
(
    SELECT genus_id, genus_name, family_id
    FROM tGenus
)

SELECT 
    family_id, 
    genus_id, 
    genus_name, 
    sample_id, 
    sum(relative_abundance) as relative_abundance
FROM species
JOIN genus USING (genus_id)
JOIN tSpecies_abundance USING (species_id, sample_id)
GROUP BY family_id, genus_id, genus_name, sample_id
ORDER BY relative_abundance DESC
;"""

vFamily = """
CREATE VIEW vFamily AS
WITH 
family AS
(
    SELECT family_id, family_name, order_id
    FROM tFamily
)

SELECT 
    order_id, 
    family_id, 
    family_name, 
    sample_id, 
    sum(relative_abundance) as relative_abundance
FROM family
JOIN vGenus USING (family_id)
GROUP BY sample_id, family_id, order_id
ORDER BY relative_abundance DESC
;"""

vOrder = """
CREATE VIEW vOrder AS
WITH 
vorder AS
(
    SELECT order_id, order_name, class_id
    FROM tOrder
)

SELECT 
    order_id, 
    order_name, 
    class_id, 
    sample_id, 
    sum(relative_abundance) as relative_abundance
FROM vorder
JOIN vFamily USING (order_id)
GROUP BY sample_id, order_id, class_id
ORDER BY relative_abundance DESC
;"""

vClass = """
CREATE VIEW vClass AS
WITH 
class AS
(
    SELECT class_id, class_name, phylum_id
    FROM tClass
)

SELECT 
    class_id, 
    class_name, 
    phylum_id, 
    sample_id, 
    sum(relative_abundance) as relative_abundance
FROM class
JOIN vOrder USING (class_id)
GROUP BY sample_id, class_id, phylum_id
ORDER BY relative_abundance DESC
;"""

vPhylum = """
CREATE VIEW vPhylum AS
WITH 
phylum AS
(
    SELECT phylum_id, phylum_name, superkingdom_id
    FROM tPhylum
)

SELECT 
    superkingdom_id, 
    phylum_id, 
    phylum_name, 
    sample_id, 
    sum(relative_abundance) as relative_abundance
FROM phylum
JOIN vClass USING (phylum_id)
GROUP BY sample_id, phylum_id, superkingdom_id
ORDER BY relative_abundance DESC
;"""


##############################################################################
##----------------------------  ENDED VIEWS  -------------------------------##
##############################################################################


def get_all_species(species_name: str = None, sample_id: str = None) -> str:
    """
    Arguments:
        species_name (str | None): name of the species to query
        sample_id (str | None): sample id else get all samples
    Returns:
        sql_statement of type `str`
    """

    if not species_name:
        species_name = ""
    if not sample_id:
        sample_id = ""

    sql = f"""
    SELECT species_id, species_name, relative_abundance, tSpecies.sample_id
    FROM tSpecies
    JOIN tSpecies_abundance USING(species_id, sample_id)
    WHERE tSpecies.species_name LIKE '{species_name}%' 
    AND tSpecies.sample_id LIKE '{sample_id}%'
    ORDER BY relative_abundance DESC
    ;"""
    return sql


def get_all_phyla(sample_id: str = None) -> str:
    """
    Calculate the relative abundance data for all phyla in the dataset.
    If no sample_id is provided, it returns data for all phyla abundance
    specific to the sample id
    Arguments:
        sample_id (str | None): sample id
    Returns:
        sql_statement (str): sql statement string
    """
    
    if not sample_id:
        sample_id = ''
    sql = f"""
    SELECT phylum_name, phylum_id, superkingdom_id, sample_id, relative_abundance
    from vPhylum
    WHERE sample_id LIKE '{sample_id}%'
    ;"""
    return sql

def get_all_class(sample_id: str = None) -> str:
    """
    Sequel string to generated to retrieve data on all class levels in a sample
    Arguments:
        sample_id (str | None): if none, all samples are returned ***yet to complete this 
    Returns:
        str: sql statement
    """
    
    if not sample_id:
        sample_id = ""
    
    sql = f"""
    SELECT class_name, class_id, sample_id, relative_abundance
    FROM vClass
    WHERE sample_id LIKE '{sample_id}%'
    ;"""
    return sql

def get_all_order(sample_id: str = None) -> str:
    """
    Arguments:
        sample_id (str): sample id to be searched.
            If none, all samples returned.
    Returns:
        str: sql statement
    """
    
    if not sample_id:
        sample_id = ""
    
    sql = f"""
    SELECT order_name, order_id, sample_id, relative_abundance
    FROM vOrder
    WHERE sample_id LIKE '{sample_id}%'
    ;"""
    return sql

def get_all_family(sample_id: str = None) -> str:
    """
    Arguments:
        sample_id (str): 
    Returns:
        str: sql statement
    """
    
    if not sample_id:
        sample_id = ""
    
    sql = f"""
    SELECT family_name, family_id, sample_id, relative_abundance
    FROM vFamily
    WHERE sample_id LIKE '{sample_id}%'
    ;"""
    return sql

def get_all_genus(sample_id: str = None) -> str:
    """
    Arguments:
        sample_id (str | None): sample id
    Returns:
        str: sql statement
    """
    
    if not sample_id:
        sample_id = ""
    sql = f"""
    SELECT genus_name, genus_id, sample_id, relative_abundance
    FROM vGenus
    WHERE sample_id LIKE '{sample_id}%'
    ;"""
    return sql
