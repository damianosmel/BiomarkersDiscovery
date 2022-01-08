from os.path import join
import pandas as pd
from utils import create_dir


class PreprocessGeneExpression:
	"""
	Class to preprocess the input csv with gene expression profile
	Inputs: gene expression profile and the metadata csv
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
		self.preprocessed_df = None
		self.out_path = join(work_root_path, "preprocess")
		create_dir(self.out_path, "preprocess output")

	def read_input_data(self, metadata_full_path, genes_data_full_path, num_first_genes):
		"""
		Read input data

		Parameters
		----------
		metadata_full_path : str
			full path for metadata xlsx file
		genes_data_full_path : str
			full path for gene expression csv file
		num_first_genes : int
			number of genes to be selected in presented order

		Returns
		-------
		metadata_df : pandas.DataFrame
			Metadata dataframe
		gene_expression_df : pandas.DataFrame
			Gene expression dataframe
		"""
		print("Reading input data, please wait..")
		metadata_df = pd.read_excel(metadata_full_path, sheet_name="Metadata")
		if num_first_genes == -1:  # all genes to be selected
			gene_expression_df = pd.read_csv(genes_data_full_path)
		else:
			gene_expression_df = pd.read_csv(genes_data_full_path, nrows=num_first_genes)
		print("Metadata: ")
		print(metadata_df.head())
		print()
		print("Gene expression: ")
		print(gene_expression_df.head())
		return metadata_df, gene_expression_df

	def merge_metadata_and_gene_expression(self, metadata_df, gene_expression_df, csv_name2save):
		"""
		Method to merge metadata xsl sheet and gene expression csv file

		Parameters
		----------
		metadata_df : pandas.DataFrame
			metadata dataframe to be used in merging

		gene_expression_df : pandas.DataFrame
			gene expression dataframe to be used in merging

		csv_name2save : str
			Name to save csv file of merged dataframe or None not to save the csv

		Returns
		-------
		metadata_gene_expression_df : pandas.DataFrame
			merged dataframe of metadata and bacteria distribution dataframes
		"""
		print("*** ***")
		print("Merging metadata and gene expression data, please wait..")
		print("Metadata num of rows, columns: {}".format(metadata_df.shape))
		print("Transpose gene expression data")

		gene_expression_transpose = gene_expression_df.set_index('Genes').transpose()
		print("Gene expression data num of rows, columns: {}".format(gene_expression_transpose.shape))

		metadata_gene_expression_df = None
		print("\n Merge metadata and transposed gene expression")
		metadata_gene_expression_df = metadata_df.merge(gene_expression_transpose, left_on="ID", right_index=True)
		print(
			"Metadata merged with gene expression: num of rows, columns: {}".format(metadata_gene_expression_df.shape))
		if csv_name2save is not None:
			print("Saving metadata merged with gene expression csv file")
			metadata_gene_expression_df.to_csv(join(self.out_path, csv_name2save), sep=",", index=False)
		return metadata_gene_expression_df
