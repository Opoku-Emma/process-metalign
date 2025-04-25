
RANK_POS = {
        "KINGDOM_POS": 0,
        "PHYLUM_POS": 1,
        "CLASS_POS": 2,
        "ORDER_POS": 3,
        "FAMILY_POS": 4,
        "GENUS_POS": 5,
        "SPECIES_POS": 6,
        "STRAIN_POS": 7,
        }

def get_kingdom(lineage: str, taxa: str) -> tuple:
    """
    Arguments:
        lineage (str): It has the format KINGDOM|PHYLUM|CLASS|ORDER|GENUS|SPECIES
        taxa (str): It has the format K_ID|P_ID|C_ID|O_ID|F_ID|G_ID|S_ID
    Returns:
        tuple: id, name"""
    kingdom_id = lineage.split('|')[RANK_POS["KINGDOM_POS"]]
    kingdom_name = taxa.split('|')[RANK_POS["KINGDOM_POS"]]
    return int(kingdom_id), kingdom_name

def get_phylum(lineage: str, taxa: str) -> tuple:
    phylum_id = lineage.split('|')[RANK_POS["PHYLUM_POS"]]
    phylum_name = taxa.split('|')[RANK_POS["PHYLUM_POS"]]
    return int(phylum_id), phylum_name

def get_class(lineage: str, taxa: str) -> tuple:
    Class_id = lineage.split('|')[RANK_POS["CLASS_POS"]]
    Class_name = taxa.split('|')[RANK_POS["CLASS_POS"]]
    return int(Class_id), Class_name

def get_order(lineage: str, taxa: str) -> tuple:
    order_id = lineage.split('|')[RANK_POS["ORDER_POS"]]
    order_name = taxa.split('|')[RANK_POS["ORDER_POS"]]
    return int(order_id), order_name

def get_family(lineage: str, taxa: str) -> tuple:
    family_id = lineage.split('|')[RANK_POS["FAMILY_POS"]]
    family_name = taxa.split('|')[RANK_POS["FAMILY_POS"]]
    return int(family_id), family_name

def get_genus(lineage: str, taxa: str) -> tuple:
    genus_id = lineage.split('|')[RANK_POS["GENUS_POS"]]
    genus_name = taxa.split('|')[RANK_POS["GENUS_POS"]]
    return int(genus_id), genus_name

def get_species(lineage: str, taxa: str) -> tuple:
    species_id = lineage.split('|')[RANK_POS["SPECIES_POS"]]
    species_name = taxa.split('|')[RANK_POS["SPECIES_POS"]]
    return int(species_id), species_name

def get_strain(lineage: str, taxa: str) -> tuple:
    try:
        strain_id = lineage.split('|')[RANK_POS["STRAIN_POS"]]
        strain_name = taxa.split('|')[RANK_POS["STRAIN_POS"]]
        return int(strain_id), strain_name
    except IndexError:
        print(f'No strain records for the lineage provided: \n{lineage}')
        
        