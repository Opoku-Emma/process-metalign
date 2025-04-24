"""
SQL create table query statements
"""

def create_tSpecies():
    sql = """
    CREATE TABLE tSpecies (
        species_id INTEGER,
        sample_id INTEGER,
        genus_id INTEGER,
        species_name TEXT,
        PRIMARY KEY (species_id, sample_id))
        ;"""
    return sql

def create_tGenus():
    sql = """
    CREATE TABLE tGenus (
        genus_id INTEGER,
        family_id INTEGER,
        genus_name TEXT,
        PRIMARY KEY (genus_id))
        ;"""
    return sql

def create_tFamily():
    sql = """
    CREATE TABLE tFamily (
        family_id INTEGER,
        order_id INTEGER,
        family_name TEXT,
        PRIMARY KEY (family_id))
        ;"""
    return sql

def create_tOrder():
    sql = """
    CREATE TABLE tOrder (
        order_id INTEGER,
        class_id INTEGER,
        order_name TEXT,
        PRIMARY KEY (order_id))
        ;"""
    return sql

def create_tClass():
    sql = """
    CREATE TABLE tClass (
        class_id INTEGER,
        phylum_id INTEGER,
        class_name TEXT,
        PRIMARY KEY (class_id))
        ;"""
    return sql

def create_tPhylum():
    sql = """
    CREATE TABLE tPhylum (
        phylum_id INTEGER,
        superkingdom_id INTEGER,
        phylum_name TEXT,
        PRIMARY KEY (phylum_id))
        ;"""
    return sql

def create_species_abundance():
    sql = """
    CREATE TABLE tSpecies_abundance (
    species_id INTEGER, 
    sample_id INTEGER, 
    relative_abundance DECIMAL(10, 5),
    PRIMARY KEY (species_id, sample_id)
    )
    ;"""
    
    return sql