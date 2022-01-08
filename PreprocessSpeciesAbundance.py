from os.path import join
import pandas as pd
from utils import create_dir


class PreprocessSpeciesAbundance:
	"""
	Class to preprocess cofactors and bacteria species abundance based on RNA NGS sequencing.
	Input file: xlsx format
	"""

	def __init__(self, work_root_path):
		"""
		Init method for class

		Parameters
		----------
		work_root_path : str
			working directory full path

		"""
		self.work_root_path = work_root_path
		# self.input_cofactors_df = None
		# self.input_bacteria_distr_df = None
		self.preprocessed_df = None
		self.out_path = join(work_root_path, "preprocess")
		create_dir(self.out_path, "preprocess output")

	def read_input_data(self, data_full_path):
		"""
		Method to read input data

		Parameters
		----------
		self : object
		Preprocess object set up for this analysis

		data_full_path : basestring
		Input data full path file

		Returns
		-------
		metadata_df : pandas.DataFrame
			Read metadata dataframe

		bacteria_distr_df : pandas.DataFrame
			Read bacteria distribution dataframe
		"""
		print("Reading input data, please wait..")
		metadata_df = pd.read_excel(data_full_path, sheet_name="Metadata")
		bacteria_distr_df = pd.read_excel(data_full_path, sheet_name="SpeciesRelativeAbundance")
		print("Metadata: ")
		print(metadata_df.head())
		print()
		print("Bacteria distribution data: ")
		print(bacteria_distr_df.head())

		return metadata_df, bacteria_distr_df

	def merge_species_and_genes(self, bacteria_distr_df, gene_expression_df, csv_name2save):
		"""
		Merge bacteria and genes dataframes

		Parameters
		----------
		bacteria_distr_df :
		gene_expression_df :

		Returns
		-------
		"""
		print("Merge bacteria and genes data")
		print("Transpose bacteria distribution data")
		# remove unused columns
		bacteria_distr_df.drop(["HOT", "Phylum", "Class", "Order", "Family", "Genus", "SpeciesShort"], axis=1,
		                       inplace=True)
		bacteria_distr_df.drop(["Unnamed: 6", "Unnamed: 7", "Unnamed: 9"], axis=1, inplace=True)
		bacteria_distr_transpose = bacteria_distr_df.set_index('Species').transpose()
		print("Bacteria distribution data num of rows, columns: {}".format(bacteria_distr_transpose.shape))
		print(bacteria_distr_transpose.head())
		print("---")

		print("Transpose gene expression data")
		gene_expression_transpose = gene_expression_df.set_index('Genes').transpose()
		# gene_expression_transpose['Genes'] = gene_expression_transpose['Genes'].str.split("PBM_")[1]
		print("Gene expression data num of rows, columns: {}".format(gene_expression_transpose.shape))
		print(gene_expression_transpose.head())
		print("---")

		merged_species_genes_df = None
		print("\n Merge transposed bacteria distribution and genes expression")
		merged_species_genes_df = gene_expression_transpose.merge(bacteria_distr_transpose, left_on="Genes",
		                                                          right_index=True)
		if csv_name2save is not None:
			print("Saving metadata merged with bacteria distribution csv file")
			merged_species_genes_df.to_csv(join(self.out_path, csv_name2save), sep=",", index=False)
		return merged_species_genes_df

	def merge_metadata_and_bacteria_distr(self, metadata_df, bacteria_distr_df, csv_name2save):
		"""
		Method to merge metadata xsl sheet and bacteria distribution sheet

		Parameters
		----------
		metadata_df : pandas.DataFrame
			metadata dataframe to be used in merging

		bacteria_distr_df : pandas.DataFrame
			bacteria distribution dataframe to be used in merging

		csv_name2save : str
			Name to save csv file of merged dataframe or None not to save the csv

		Returns
		-------
		metadata_bacteria_dist_df : pandas.DataFrame
			merged dataframe of metadata and bacteria distribution dataframes
		"""
		print("*** ***")
		print("Merging metadata and bacteria distribution data, please wait..")
		print("Metadata num of rows, columns: {}".format(metadata_df.shape))
		print("Transpose bacteria distribution data")
		# remove unused columns
		bacteria_distr_df.drop(["HOT", "Phylum", "Class", "Order", "Family", "Genus", "SpeciesShort"], axis=1,
		                       inplace=True)
		bacteria_distr_df.drop(["Unnamed: 6", "Unnamed: 7", "Unnamed: 9"], axis=1, inplace=True)
		bacteria_distr_transpose = bacteria_distr_df.set_index('Species').transpose()
		print("Bacteria distribution data num of rows, columns: {}".format(bacteria_distr_transpose.shape))

		metadata_bacteria_dist_df = None
		print("\n Merge metadata and transposed bacteria distribution")
		metadata_bacteria_dist_df = metadata_df.merge(bacteria_distr_transpose, left_on="SampleNo.", right_index=True)
		print("Metadata merged with species: num of rows, columns: {}".format(metadata_bacteria_dist_df.shape))
		if csv_name2save is not None:
			print("Saving metadata merged with bacteria distribution csv file")
			metadata_bacteria_dist_df.to_csv(join(self.out_path, csv_name2save), sep=",", index=False)
		return metadata_bacteria_dist_df
