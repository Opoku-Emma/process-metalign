from src.metalign_analysis import MetalignDB as DB

def main():
    raw_data = "data/Leaf_all.nostrain.txt"
    db = DB(raw_data)
    metadata_file = "data/leaf_phenotype.csv"
    db.get_metadata(metadata_file, categories=['samples', 'site', 'treatment_herb_level'], sep=',', index_col='samples')
    db.plot_pcoa(color_by="site", method='eigh')  
    return

if __name__ == "__main__":
    main()
        
        
