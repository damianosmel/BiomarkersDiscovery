# Biomarkers discovery for dental implants

Repository for project *biomarker discovery for dental implants!*

The code preprocesses two kinds of biological data and applies machine learning techniques to predict dental implant outcome

## Main dependencies
The main dependencies are listed below:
* Python 3.7.8
* Pandas 1.1.4
* Numpy 1.19.4
* Matplotlib 3.1.1
* scikit-learn 0.22.1
* stability_selection from: [patch_github](https://github.com/scikit-learn-contrib/stability-selection)

For R script that runs ridge and lasso projection technique, you will need:
* hdi
* ggplot2
* reshape2

## Data
This repository handles two kinds of data:
* bacterial species abundance
* gene expression data

Species and gene data are found in [data](data) folder.

## Biomarkers discovery
Biomarkers discovery was formed as the following problem: given as input features, the bacterial species abundance or gene expression per dental implant sample, find **the most informative features subset** to predict a two or three class problem.
The two-class problem is to distinguish between health and peri-implantitis (label: diagnosis1). The three-class problem is distinguish between health, mucositis and peri-implantitis (label: diagnosis3). 

Identification of biomarkers was performed by Dr. Tuan-Anh Hong and Msc. Damianos P. Melidis.

### Tuan-Anh's work
Tuan-Anh has performed brute-force combination of species abundance to find the most performing subset for the two-class problem. 

Please see the [tuanAnh](tuanAnh) folder, for his code and report.

### Damianos' work
I have performed embedded methods to perform feature selection as part of the prediction optimization problem. I have used the following methods:
* stability selection
* ridge and lasso projection (using the R [script](biomarker_disco.R))
* union of lasso selected features subset, each run through for one fold of a cross validation procedure

### Bacterial species abundance and dental implants
To perform selection of bacterial sepcies for dental implant two or three-class problem:
 * go to [species_abundance_pred_run](species_abundance_pred_run.py)
 * update the data paths variables: `work_root_path`, `data_full_path`
 * do not change the `merged_csv_name` variable
 * to set up the biomarker discovery, examine the run() function in [Models](Models.py) class
 * then set up the variables found under the *Step 2 - Classify input data*
 
### Gene expression and dental implants
 * go to [genes_expression_pred_run](biomarker_disco/genes_expression_pred_run.py)
 * update the data paths: `work_root_path`, `metadata_full_path`, `gene_data_full_path`
 * select the number of genes that you will use as input features, set up the variable: `num_first_genes`
 * to set up the biomarker discovery, examine the run() function in [Models](Models.py) class
 * then set up the variables found undert the *Step 2 - Classify input data*
 
### Input Features selected by R code to Python code
 * run the R [script](biomarker_disco.R)
 * then create a .txt file containing each of the selected features per line.
 For example, see [selected_features_lasso_projection](selected_features/filter_lasso_projection_diagnosis1.txt) file.
 * then set up the variable `preselected_features_file` with the full path of the selected features file, on the previous two pred_run.py scripts 

For all selected features so far, please see [selected_features](selected_features) folder.
