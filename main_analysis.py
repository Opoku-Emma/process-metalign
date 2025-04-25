from src.metalign_analysis import MetalignDB as DB

def main():
    rurutu_soil = "data/all_samples_nostrain.tsv"
    db = DB(rurutu_soil)
    metadata_file = "data/RurutuSoilNewTemplate(1).txt"
    db.get_metadata(metadata_file, categories=['Well', 'Sample Name','complex', 'cultivation'], index_col='Well')
    # alpha_diversity = db.get_alpha_diversity()
    # print(f"Alpha diversity for all samples: \n{list(alpha_diversity)}")
    db.plot_pcoa('complex')
    return

if __name__ == "__main__":
    main()
        
        
