library(hdi)
library(ggplot2)
library(reshape2)
### ### ###
# set up current directory
# read data set
### ### ###
data_path <- "/home/damian/Desktop/biomarker_discovery/preprocess"
# server
#data_path <- "/home/melidis/bac_data/biomarker_discovery/preprocess" 
setwd(data_path)
# raw_data <- read.csv("cofactors_all_species.csv")
raw_data <- read.csv("metadata_1000_genes.csv")
### ### ###
# get X,Y
# change diagnosis3 column (peri-implanitis, non-peri-implantitis)
# get the design matrix
### ### ###
cat("Converting diagnosis 3 column: non-peri-implantitis -> 0, peri-implantitis -> 1.\n")
# data_processed_y <- transform(raw_data, Diagnosis3 = c(0,1)[as.numeric(Diagnosis3)]) #diagnosis3
data_processed_y <- transform(raw_data, Diagnosis1 = c(0,1,2)[as.numeric(Diagnosis1)])
cat("Extract Y label <- Diagnosis3 or Diagnosis1 column")
# Y_diagnosis <- data_processed_y[,c("Diagnosis3")] # Diagnosis3
Y_diagnosis <- data_processed_y[,c("Diagnosis1")]
cat("Get design matrix X <- species abundance")
whole_dataframe_shape <- dim(data_processed_y) #get the number columns of the whole dataframe
X_raw <- data.matrix(data_processed_y[,45:whole_dataframe_shape[2]])
X_maxs <- apply(X_raw, 2, max)
X_mins <- apply(X_raw, 2, min)
X_species_abund <- scale(X_raw, center=X_mins,scale=X_maxs-X_mins)

cat("Check the dimensions: ")
cat("X species abundance dimensions:")
cat(dim(X_species_abund))
# cat("Y, non-peri-implantitis -> 0, peri-implantitis -> 1, dimensions: ")
cat("Y, health -> 0, mucositis -> 1, peri-implantitis -> 2, dimensions: ")
cat(dim(Y_diagnosis))

### ### ###
# Apply variable selection, using ridge projection technique
### ### ###
cat("Apply ridge projection")
# out_ridge <- ridge.proj(x=X_species_abund,y=Y_diagnosis,family="binomial") # diagnosis3
out_ridge <- ridge.proj(x=X_species_abund,y=Y_diagnosis,family="gaussian")
cat("p-values: ")
cat(out_ridge$pval)
cat("corrected p-values: ")
cat(out_ridge$pval.corr)
conf_intervals_95_perc <- confint(out_ridge,parm= which(out_ridge$pval.corr <= 0.05), level=0.95)

### ### ###
# Apply variable selection, using lasso projection technique
### ### ###
cat("Apply lasso projection")
# out_lasso <- lasso.proj(x=X_species_abund,y=Y_diagnosis,family="binomial") #diagnosis3
out_lasso <- lasso.proj(x=X_species_abund,y=Y_diagnosis,family="gaussian")
cat("p-values: ")
cat(out_lasso$pval)
cat("corrected p-values: ")
cat(out_lasso$pval.corr)
conf_intervals_95_perc <- confint(out_lasso,parm=which(out_lasso$pval.corr <= 0.05),level=0.95)
cat(conf_intervals_95_perc)
cat(which(out_lasso$pval.corr <= 0.1))
#diagnosis3 - species
#streptococcus_sp056_pval_corr <- out_lasso$pval.corr["Streptococcus_sp_056_HOT_56"]
#streptococcus_sp066_pval_corr <- out_lasso$pval.corr["Streptococcus_sp_066_HOT_66"]

### ### ###
# Plot these species abundance
### ### ###
# X_Y <- as.data.frame(X_species_abund)
# X_Y["diagnosis3"] <- Y_diagnosis3
# ggplot(as.data.frame(X_Y), aes(x=Streptococcus_sp_056_HOT_56, y=Streptococcus_sp_066_HOT_66,color=factor(diagnosis3))) + geom_point()
cat("### ###")
