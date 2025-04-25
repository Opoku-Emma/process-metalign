import re 
import src.preprocess_data as preprocess_data


def get_sample_name(sample_id: str) -> str:
    """
    Extract sample name. It literally just prints the Well number
    The regex can be modified to print out different values from the text
    Arguments:
        sample_id (str): the sample id
    Returns:
        str: well number that matches to sample id
    """
    regex = re.compile(
        r"^[A-Za-z0-9]{0,}_"
        r"([A-Z0-9a-z]{1,})_"
        r"([A-Z0-9a-z]{1,}).*"
    )
    results = regex.findall(sample_id)
    return results[0]


def make_row_dict(lineage: str, taxon: str, sample_id: str) -> dict:
    """
    Create a dictionary of taxonomic information based on lineage, taxon, and sample ID.

    Arguments:
        lineage (str): The lineage string.
        taxon (str): The taxon name.
        sample_id (str): The sample ID.
    Returns:
        dict: A dictionary containing taxonomic information.
    """
    rows_dict = {}
    sample_id = get_sample_name(sample_id)[0] # Substring of the sample id. This is just enough for them to be unique

    kingdom_id, kingdom_name =   preprocess_data.get_kingdom(lineage, taxon)
    phylum_id, phylum_name  =    preprocess_data.get_phylum(lineage, taxon)
    Class_id, class_name    =    preprocess_data.get_class(lineage, taxon)
    order_id, order_name    =    preprocess_data.get_order(lineage, taxon)
    family_id, family_name  =    preprocess_data.get_family(lineage, taxon)
    genus_id, genus_name    =    preprocess_data.get_genus(lineage, taxon)
    species_id, species_name =   preprocess_data.get_species(lineage, taxon)

    rows_dict['kingdom'] = [kingdom_id, kingdom_name]
    rows_dict['phylum'] = [phylum_id, kingdom_id, phylum_name]
    rows_dict['class'] = [Class_id, phylum_id, class_name]
    rows_dict['order'] = [order_id, Class_id, order_name]
    rows_dict['family'] = [family_id, order_id, family_name]
    rows_dict['genus'] = [genus_id, family_id, genus_name]
    rows_dict['species'] = [species_id, sample_id, genus_id, species_name]

    return rows_dict
