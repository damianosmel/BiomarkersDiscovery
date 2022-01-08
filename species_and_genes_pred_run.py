from Models import Models
from PreprocessSpeciesAbundance import PreprocessSpeciesAbundance
from PreprocessGeneExpression import PreprocessGeneExpression


###                                             ###
### Predict clinical outcome of dental implant  ###
### based on species abundance                  ###
### and gene expression data                    ###
###                                             ###


###                                 ###
### Step 1 - Preprocess input data  ###
###                                 ###

# species data
print(" === Preprocess input data === ")
work_root_path = "/home/damian/Desktop/biomarker_discovery"
metadata_full_path = "/home/damian/Desktop/biomarker_discovery/species_abundance/FirstDataSet.xlsx"

Preprocess_species_abundance = PreprocessSpeciesAbundance(work_root_path)
metadata_df, bacteria_distr_df = Preprocess_species_abundance.read_input_data(metadata_full_path)

# genes data
gene_data_full_path = "/home/damian/Desktop/biomarker_discovery/genes_expression/sorted_HOMD_geneHitsv19_1M_normalized.csv"
Preprocess_gene_expression = PreprocessGeneExpression(work_root_path)
metadata_df, gene_expression_df = Preprocess_gene_expression.read_input_data(metadata_full_path, gene_data_full_path)
merged_csv_name = "species_genes.csv"
species_genes_df = Preprocess_species_abundance.merge_species_and_genes(bacteria_distr_df,gene_expression_df,merged_csv_name)