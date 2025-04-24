def assign_code(lineage: str, taxon: str) -> str:
    """Check lineages and assign codes to those without data
    Arguments:
        lineage (str): a string of format `A|B|C|D|E|F`
        taxon (str): a string of format `A|B|C|D|E|F`
    Returns:
        str
    """
    
    # Reverse so that we start from species level
    lineage = lineage.split('|')[::-1]
    taxon = taxon.split('|')[::-1]
    
    prefixes = {'genus': '-10', 'family': '-20', 'order': '-30', 'class': '-40', 'phylum': '-50'}
    hierarchy = ['genus', 'family', 'order', 'class', 'phylum']

    for index, level in enumerate(hierarchy, 1):
        # Since index is starting at 1, it will skip the species in the lineage
        if not lineage[index]:
            if lineage[index - 1].startswith('-'):
                lineage[index] = lineage[index - 1]
                taxon[index] = taxon[index - 1]
            else:
                lineage[index] = prefixes[level] + lineage[index - 1]
                taxon[index] = 'NA_' + taxon[index - 1]  
    if (x:=lineage[-1]) in ('', ' ', None):
        lineage[-1] = '-2222'
        taxon[-1] = 'Unknown_Kingdom'
    
    return "|".join(lineage[::-1]), "|".join(taxon[::-1])
