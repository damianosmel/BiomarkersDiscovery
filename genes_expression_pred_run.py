from PreprocessGeneExpression import PreprocessGeneExpression
from Models import Models
from os.path import join
from pandas import read_csv

###                                             ###
### Predict clinical outcome of dental implant  ###
### based on gene expression data               ###
###                                             ###

###                                 ###
### Step 1 - Preprocess input data  ###
###                                 ###
print(" === Preprocess input data === ")

### local machine ###
work_root_path = "/home/damian/Desktop/biomarker_discovery"
metadata_full_path = "/home/damian/Desktop/biomarker_discovery/species_abundance/FirstDataSet.xlsx"
gene_data_full_path = "/home/damian/Desktop/biomarker_discovery/genes_expression/sorted_HOMD_geneHitsv19_1M_normalized.csv"
num_first_genes = 1000
Preprocess_gene_expression = PreprocessGeneExpression(work_root_path)
metadata_df, gene_expression_df = Preprocess_gene_expression.read_input_data(metadata_full_path, gene_data_full_path,
                                                                             num_first_genes)

if num_first_genes == -1:
	merged_csv_name = "metadata_all_genes.csv"
else:
	merged_csv_name = "metadata_" + str(num_first_genes) + "_genes.csv"
merged_df = Preprocess_gene_expression.merge_metadata_and_gene_expression(metadata_df, gene_expression_df,
                                                                          merged_csv_name)

###                                 ###
### Step 2 - Classify input data    ###
###                                 ###
print("\n === Classify data === ")
### local machine ###
cofactors_type = "gene_expression"  # can be ["species_abundance", "gene_expression", "species_and_genes"]
target_label = "Diagnosis1"
classifier_name = "logistic_l2"  # "svm" #"tree" #"logistic_l1"
create_test_split = False  # use cross-validation
# For Lasso, normalization can be done from sci-kit learn, so no need
normalize_X = True  # normalize X data
apply_stability_selection = False  # apply stability selection to select features
preselected_features_file = None
# preselected_features_file = "/home/damian/Desktop/biomarker_discovery/genes_diagnosis1_lasso_proj_pval01.txt"
preselected_features_file = "/home/damian/Desktop/biomarker_discovery/logistic_l1_features.txt"
preselected_features_file = "/home/damian/Desktop/biomarker_discovery/genes_diagnosis1_logistic_l1_features.txt"
classifier = Models(work_root_path, merged_df)
classifier.run(cofactors_type, target_label, classifier_name, normalize_X, create_test_split,
               apply_stability_selection, preselected_features_file)

#
# ### server ###
# # work_root_path = "/home/melidis/bac_data/biomarker_discovery"
# # preprocess_path = join(work_root_path, "preprocess")
# # merged_df = read_csv(join(preprocess_path, "metadata_all_genes.csv"))
# # cofactors_type = "gene_expression"
# # create_test_split = False
# # normalize_X = True
#
#
###                           ###
### Run models for diagnosis3 ###
###                           ###
# print(" ### Diagnosis3 ### ")
# target_label = "Diagnosis3"
#
# print(" ### SVM no preselected features ### ")
# classifier_name = "svm"
# apply_stability_selection = False
#
# preselected_features_file = None
#
# svm_diagnosis3 = Models(work_root_path, merged_df)
# svm_diagnosis3.run(cofactors_type, target_label, classifier_name, normalize_X, create_test_split,
#                    apply_stability_selection, preselected_features_file)
#
# print(" ### Logistic L2 no preselected features ### ")
# classifier_name = "logistic_l2"
# apply_stability_selection = False
# preselected_features_file = None
#
# svm_diagnosis3 = Models(work_root_path, merged_df)
# svm_diagnosis3.run(cofactors_type, target_label, classifier_name, normalize_X, create_test_split,
#                    apply_stability_selection, preselected_features_file)
#
# print(" ### Logistic L1 no preselected features ### ")
# classifier_name = "logistic_l1"
# apply_stability_selection = False
# preselected_features_file = None
#
# svm_diagnosis3 = Models(work_root_path, merged_df)
# svm_diagnosis3.run(cofactors_type, target_label, classifier_name, normalize_X, create_test_split,
#                    apply_stability_selection, preselected_features_file)
#
# print(" ### Decision tree no preselected features ### ")
# classifier_name = "tree"
# apply_stability_selection = False
# preselected_features_file = None
#
# svm_diagnosis3 = Models(work_root_path, merged_df)
# svm_diagnosis3.run(cofactors_type, target_label, classifier_name, normalize_X, create_test_split,
#                    apply_stability_selection, preselected_features_file)
#
# print(" ### Logistic L2 no preselected features ### ")
# classifier_name = "logistic_l2"
# apply_stability_selection = True
# preselected_features_file = None
#
# svm_diagnosis3 = Models(work_root_path, merged_df)
# svm_diagnosis3.run(cofactors_type, target_label, classifier_name, normalize_X, create_test_split,
#                    apply_stability_selection, preselected_features_file)
# print(" --- * --- ")
#
# ###                           ###
# ### Run models for diagnosis3 ###
# ###                           ###
# print("\n ### Diagnosis1 ### ")
# target_label = "Diagnosis1"
#
# print(" ### SVM no preselected features ### ")
# classifier_name = "svm"
# apply_stability_selection = False
#
# preselected_features_file = None
#
# svm_diagnosis1 = Models(work_root_path, merged_df)
# svm_diagnosis1.run(cofactors_type, target_label, classifier_name, normalize_X, create_test_split,
#                    apply_stability_selection, preselected_features_file)
#
# print("\n ### Logistic L2 no preselected features ### ")
# classifier_name = "logistic_l2"
# apply_stability_selection = False
# preselected_features_file = None
#
# svm_diagnosis1 = Models(work_root_path, merged_df)
# svm_diagnosis1.run(cofactors_type, target_label, classifier_name, normalize_X, create_test_split,
#                    apply_stability_selection, preselected_features_file)
#
# print("\n ### Logistic L1 no preselected features ### ")
# classifier_name = "logistic_l1"
# apply_stability_selection = False
# preselected_features_file = None
#
# svm_diagnosis1 = Models(work_root_path, merged_df)
# svm_diagnosis1.run(cofactors_type, target_label, classifier_name, normalize_X, create_test_split,
#                    apply_stability_selection, preselected_features_file)
#
# print("\n ### Decision tree no preselected features ### ")
# classifier_name = "tree"
# apply_stability_selection = False
# preselected_features_file = None
#
# svm_diagnosis1 = Models(work_root_path, merged_df)
# svm_diagnosis1.run(cofactors_type, target_label, classifier_name, normalize_X, create_test_split,
#                    apply_stability_selection, preselected_features_file)
#
# print("\n ### Logistic L2 no preselected features ### ")
# classifier_name = "logistic_l2"
# apply_stability_selection = True
# preselected_features_file = None
#
# svm_diagnosis1 = Models(work_root_path, merged_df)
# svm_diagnosis1.run(cofactors_type, target_label, classifier_name, normalize_X, create_test_split,
#                    apply_stability_selection, preselected_features_file)
# print(" --- * --- ")
