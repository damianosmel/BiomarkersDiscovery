from Models import Models
from PreprocessSpeciesAbundance import PreprocessSpeciesAbundance

###                                             ###
### Predict clinical outcome of dental implant  ###
### based on species abundance data             ###
###                                             ###

###                                 ###
### Step 1 - Preprocess input data  ###
###                                 ###

print(" === Preprocess input data === ")
work_root_path = "/home/damian/Desktop/biomarker_discovery"
data_full_path = "/home/damian/Desktop/biomarker_discovery/species_abundance/FirstDataSet.xlsx"

Preprocess_species_abundance = PreprocessSpeciesAbundance(work_root_path)
metadata_df, bacteria_distr_df = Preprocess_species_abundance.read_input_data(data_full_path)
merged_csv_name = "metadata_all_species.csv"
merged_df = Preprocess_species_abundance.merge_metadata_and_bacteria_distr(metadata_df, bacteria_distr_df,
                                                                           merged_csv_name)

###                                 ###
### Step 2 - Classify input data    ###
###                                 ###
print("\n === Classify data === ")
cofactors_type = "species_abundance"  # can be ["species_abundance", "gene_expression", "species_and_genes"]
target_label = "Diagnosis3"  # "Diagnosis1"
classifier_name = "tree"  # "svm" #"logistic_l1" #"tree"
create_test_split = False  # use cross-validation
# For Lasso, normalization can be done from sci-kit learn, so no need
normalize_X = True  # normalize X data
apply_stability_selection = False  # apply stability selection to select features

### preselected features ###
# if not supplying preselected features: file=None
# preselected_features_file = None
# diagnosis 3, species
# preselected_features_file = "/home/damian/Desktop/biomarker_discovery/filter_species_diagnosis1.txt"
preselected_features_file = "/home/damian/Desktop/biomarker_discovery/selected_features/lasso_projection_species_diagnosis3.txt"
# diagnosis 1, species
# preselected_features_file = "/home/damian/Desktop/biomarker_discovery/lasso_projection_species_diagnosis1_pval005.txt"
# preselected_features_file = "/home/damian/Desktop/biomarker_discovery/lasso_projection_species_diagnosis1_pval01.txt"
# preselected_features_file = "/home/damian/Desktop/biomarker_discovery/lasso_projection_species_diagnosis1_pval03.txt"
# preselected_features_file = "/home/damian/Desktop/biomarker_discovery/selected_species_all.txt"
# preselected_features_file = "/home/damian/Desktop/biomarker_discovery/l1_diagnosis1.txt"

# Tuan-Anh's selected features using filter methods
# preselected_features_file = "/home/damian/Desktop/biomarker_discovery/filter_species_2.txt"
# preselected_features_file = "/home/damian/Desktop/biomarker_discovery/filter_species_6_1.txt"
# preselected_features_file = "/home/damian/Desktop/biomarker_discovery/filter_species_6_2.txt"
# preselected_features_file = "/home/damian/Desktop/biomarker_discovery/filter_species_6_3.txt"
# preselected_features_file = "/home/damian/Desktop/biomarker_discovery/filter_species_6_4.txt"

# combine Tuan-Anh's selected features and lasso projection features
# preselected_features_file = "/home/damian/Desktop/biomarker_discovery/filter_lasso_projection_diagnosis1.txt"
# preselected_features_file = "/home/damian/Desktop/biomarker_discovery/filter_lasso_projection_diagnosis3.txt"

ClassificationModel = Models(work_root_path, merged_df)
ClassificationModel.run(cofactors_type, target_label, classifier_name, normalize_X, create_test_split,
                        apply_stability_selection, preselected_features_file)
