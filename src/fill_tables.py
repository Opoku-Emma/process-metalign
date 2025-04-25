def fill_species() -> str:
    """
    Generates an SQL query string to insert data into the tSpecies table.
    The query is designed to insert values for species_id, sample_id, genus_id, 
    and species_name into the tSpecies table.
    Returns:
        str: An SQL INSERT statement.
    """
    
    sql = """
    INSERT INTO tSpecies (species_id, sample_id, genus_id, species_name)
    VALUES (?, ?, ?, ?)
    ;"""
    return sql

def fill_genus():
    """
    Generates an SQL query string to insert data into the tGenus table.
    The query is designed to insert values for genus_id, family_id, and genus_name
    into the tGenus table.
    Returns:
        str: An SQL INSERT statement.
    """
    sql = """
    INSERT INTO tGenus (genus_id, family_id, genus_name)
    VALUES (?, ?, ?)
    ;"""
    return sql

def fill_family():
    sql = """
    INSERT INTO tFamily (family_id, order_id, family_name)
    VALUES (?, ?, ?)
    ;"""
    return sql

def fill_order():
    sql = """
    INSERT INTO tOrder (order_id, class_id, order_name)
    VALUES (?,?, ?)
    ;"""
    return sql

def fill_class():
    sql = """
    INSERT INTO tClass (class_id, phylum_id, class_name)
    VALUES (?, ?, ?)
    ;"""
    sql
    return sql

def fill_phylum():
    sql = """
    INSERT INTO tPhylum (phylum_id, superkingdom_id, phylum_name)
    VALUES (?, ?, ?)
    ;"""
    return sql

def species_abundance_sql():
    sql = """
    INSERT INTO tSpecies_abundance (species_id, sample_id, relative_abundance)
    VALUES (?,?,?)
    ;"""
    return sql
