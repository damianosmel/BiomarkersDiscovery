from ntpath import basename
from os.path import join

import numpy as np
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from stability_selection import StabilitySelection
import matplotlib.pyplot as plt

from utils import create_dir


class Models:
	"""
	Classification models to predict dental implants disease label
	based input data
	"""

	def __init__(self, work_root_path, data_df):
		"""
		Constructor for classification models class

		Parameters
		----------
		work_root_path : str
			working directory full path
		data_df : pandas.DataFrame
			input dataframe for learning
		"""
		self.work_root_path = work_root_path
		self.data_df = data_df
		self.out_path = join(self.work_root_path, "classification_perf")

		create_dir(self.out_path, "output")

		self.dataset_name = None
		self.X_train_df, self.X_test_df = None, None
		self.X_train, self.X_test = None, None
		self.y_train, self.y_test = None, None
		self.random_state = 314

	def split_train_test_stratified(self, label_name, test_portion):
		"""
		Split data into train and test using stratified split based on Y label

		Parameters
		----------
		label_name : str
			column name defining Y
		test_portion : float
			split portion of test

		Returns
		-------
		dataset_train : pandas.DataFrame
			train split of data set
		dataset_test : pandas.DataFrame
			test split of data set
		"""
		print("Apply train/test stratified split for {}, test size: {}.".format(label_name, test_portion))
		train_file_name = self.dataset_name + "_train.csv"
		test_file_name = self.dataset_name + "_test.csv"

		dataset_train, dataset_test = train_test_split(self.data_df, test_size=test_portion,
		                                               stratify=self.data_df[label_name])
		print("Writing: ")
		print("dataset_train: {}".format(dataset_train.shape))
		print("dataset_test: {}".format(dataset_test.shape))
		dataset_train.to_csv(join(self.out_path, train_file_name), sep=",", index=False)
		dataset_test.to_csv(join(self.out_path, test_file_name), sep=",", index=False)
		print("---")
		return dataset_train, dataset_test

	def normalize_X(self):
		"""
		Normalize design matrix X

		Returns
		-------
		X_train_norm : pandas.DataFrame
			normalized X train
		X_test_norm : pandas.DataFrame
			normalized X test
		"""
		# normalize data set
		print("Normalize design matrix X")
		normal_scaler = StandardScaler()
		X_train_norm = normal_scaler.fit_transform(self.X_train_df.values)
		if self.X_test_df is not None:
			X_test_norm = normal_scaler.fit_transform(self.X_test_df.values)
		else:
			X_test_norm = None
		return X_train_norm, X_test_norm

	def subset_X(self, selected_features):
		"""
		Subset X based on specified selected features

		Parameters
		----------
		selected_features : list of str
			selected features names

		Returns
		-------
		pandas.DataFrame
			selected subset of design matrix X train
		pandas.DataFrame
			selected subset of design matrix X test
		"""
		print("Subset X to contain only the {} specified selected features".format(len(selected_features)))

		# subset train split
		X_train_subset_df = self.X_train_df.loc[:, selected_features]
		assert X_train_subset_df.shape[1] == len(
			selected_features), "AssertionError: Subset of training data frame should contain only the selected features."

		# subset test split
		if self.X_test_df is not None:
			X_test_subset_df = self.X_test_df.loc[:, selected_features]
			assert X_test_subset_df.shape[1] == len(
				selected_features), "AssertionError: Subset of testing data frame should contain only selected features."
		else:
			X_test_subset_df = None
		return X_train_subset_df, X_test_subset_df

	def load_Xy(self, label_name, create_test_set):
		"""
		Load X and y for running classification model

		Parameters
		----------
		label_name : str
			label name used as target variable for learning (y)
		create_test_set : bool
			create separate test set for evaluation (yes), otherwise do not split into separate train and test sets

		Returns
		-------
		y_names : list of str
			target variable categorical names
		"""
		print("Load X and y and split (if needed)")
		# set data frame index to sample number
		self.data_df = self.data_df.drop_duplicates(subset='SampleNo.')
		self.data_df.set_index('SampleNo.', inplace=True)

		if create_test_set:
			# split into separate train and test sets
			dataset_train, dataset_test = self.split_train_test_stratified(label_name, 0.3)
		else:
			dataset_train, dataset_test = self.data_df, None
		# get index of last clinical column
		last_clinical_column_index = self.data_df.columns.get_loc("Implant type")
		# get species abundance/gene expression cofactors as X
		self.X_train_df = dataset_train.iloc[:, last_clinical_column_index + 1:self.data_df.shape[1]]
		# self.X_train_df = self.X_train_df.loc[:, ['Streptococcus_sp_056_HOT_56', 'Streptococcus_sp_066_HOT_66']]

		if create_test_set:
			self.X_test_df = dataset_test.iloc[:, last_clinical_column_index + 1:self.data_df.shape[1]]

		# get Y from selected column label
		print("Converting categorical target Y to numerical")
		le = LabelEncoder()
		le.fit_transform(dataset_train[label_name])
		self.y_train = le.transform(dataset_train[label_name])
		if create_test_set:
			self.y_test = le.transform(dataset_test[label_name])
		y_names = []
		for numerical_value, categorical_value in enumerate(list(le.classes_)):
			print("categorical= {} <-> numerical= {}".format(categorical_value, numerical_value))
			y_names.append(categorical_value)

		print("### Train ###")
		print("X= {}".format(self.X_train_df.head()))
		print("y= {}".format(self.y_train))
		if create_test_set:
			print("### Test ###")
			print("X= {}".format(self.X_test_df.head()))
			print("y= {}".format(self.y_test))
		print(" === === ")
		return y_names

	def prepare_Xy(self, normalize_X, label_name, create_test_set, apply_stability_selection,
	               preselected_features_file):
		"""
		Prepare X and y by:
		* loading data
		* apply feature selection method or subset based on preselected features
		* normalize features

		Parameters
		----------
		normalize_X : bool
			normalize design matrix X
		label_name : str
			label name used as target variable for learning (y)
		create_test_set : bool
			create separate test set for evaluation (yes), otherwise do not split into separate train and test sets
		apply_stability_selection : bool
			to apply stability selection to select features (True), otherwise use all feature set (False)
		preselected_features_file : str
			full path to file containing preselected features

		Returns
		-------
		list of str
			selected feature names, if feature selection method is selected or None if no feature selection method is used
		list of str
			categorical values of y
		"""
		print("Prepare X,y values")
		# 1) load X and y data, split in train and test (if needed)
		y_names = self.load_Xy(label_name, create_test_set)

		# 2) apply feature selection (stability selection) or use preselected features
		if apply_stability_selection:
			selected_feature_names = self.stability_selection_run()
			self.X_train_df, self.X_test_df = self.subset_X(selected_feature_names)
		elif preselected_features_file is not None:  # if user supplied preselected features load them
			selected_feature_names = self.load_preselected_features(preselected_features_file)
			# preselected_feature_indices = self.convert_feature_names2indices(preselected_feature_names)
			self.X_train_df, self.X_test_df = self.subset_X(selected_feature_names)
		else:
			selected_feature_names = None

		# 3) normalize design matrix X
		if normalize_X:
			self.X_train, self.X_test = self.normalize_X()
		else:
			self.X_train = self.X_train_df.values
			if create_test_set:
				self.X_test = self.X_test_df.values
		return selected_feature_names, y_names

	def create_out_data_folder(self, biomarkers_name):
		"""
		Create output folder specific for input data type

		Parameters
		----------
		biomarkers_name : str
			biomarkers type name

		Returns
		-------
		None
		"""
		self.out_path = join(self.out_path, biomarkers_name)
		create_dir(self.out_path, "dataset output")

	def stability_selection_run(self):
		"""
		Run stability selection to select features before running classification
		Credits: https://github.com/scikit-learn-contrib/stability-selection

		Returns
		-------
		selected_features : list of str
			resulted selected features
		"""
		print("Apply stability selection as feature selection step")
		# for the feature selection all available data are used
		X, y = self.X_train_df.values, self.y_train
		logistic_l1 = LogisticRegression(solver='liblinear', penalty='l1')
		selector = StabilitySelection(base_estimator=logistic_l1, lambda_name='C',
		                              lambda_grid=np.logspace(-15, 1, 100, endpoint=False))
		selector.fit(X, y)
		selected_features = selector.get_support(indices=True)
		selection_method = "stability_selection"
		selected_feature_names = self.find_selected_features_names(selected_features, selection_method, save_names=True)
		return selected_feature_names

	def load_preselected_features(self, preselected_features_file):
		"""
		Load preselected features from file

		Parameters
		----------
		preselected_features_file : str
			full path to preselected features file
		Returns
		-------
		preselected_feature_names : list of str
			preselected feature names
		"""
		print("Load preselected features from {}".format(preselected_features_file))
		with open(preselected_features_file) as f:
			file_content = f.readlines()
		preselected_feature_names = [feature_line.strip() for feature_line in file_content]
		print("Loaded preselected features:\n{}".format("\n".join(preselected_feature_names)))
		return preselected_feature_names

	def convert_feature_names2indices(self, selected_feature_names):
		"""
		Convert feature names to feature indices of design matrix X

		Parameters
		----------
		selected_feature_names : list of str
			list of selected features

		Returns
		-------
		list of int
			list of indices of selected features
		"""
		print("Convert feature names to feature indices of design matrix X")
		return [list(self.X_train_df.columns).index(feature_name) for feature_name in selected_feature_names]

	def run(self, biomarkers_type, label_name, model_name, normalize_X, create_test_set, apply_stability_selection,
	        preselected_features_file):
		"""
		Function to run classification model

		Parameters
		----------
		biomarkers_type : str
			biomarkers type name e.g species (abundance)
		label_name : str
			 target label name
		model_name : str
			classification model e.g. lasso
		normalize_X : bool
			normalize X input data (True), otherwise do not normalize (False)
		create_test_set : bool
			to create test set (True), otherwise don't create test set (False)
		apply_stability_selection : bool
			to apply stability selection to select features (True), otherwise use all feature set (False)
		preselected_features_file : str
			full path to file containing preselected features

		Returns
		-------
		None
		"""
		print(" ### classification start ### ")
		print("Preparing data set, classifier and then run classification")
		assert label_name in list(
			self.data_df.columns), "AssertionError: label name, {}, should be existing in the input data frame".format(
			label_name)
		# get model name
		model = model_name.split("_")[0]

		# create current run output folder
		if preselected_features_file is not None:
			feature_suffix = basename(preselected_features_file).split(".")[0]
		elif apply_stability_selection:
			feature_suffix = "stability_selection"
		else:
			feature_suffix = "no_preselected_features"
		self.create_out_data_folder(
			biomarkers_type + "_" + model_name + "_" + label_name.lower() + "_" + feature_suffix)
		self.dataset_name = biomarkers_type + "_" + label_name

		# load and split X,y, subset based on feature selection, finally normalize X
		selected_feature_names, y_names = self.prepare_Xy(normalize_X, label_name, create_test_set,
		                                                  apply_stability_selection, preselected_features_file)

		# set flag for binary or multi-class classification
		if label_name.lower() == "diagnosis3":
			is_binary_classification = True
		else:
			is_binary_classification = False

		if model == "logistic":
			regression_penalty = model_name.split("_")[1]
			assert regression_penalty in ["l2",
			                              "l1"], "AssertionError: {} penalty option unknown please keep l1 or l2".format(
				regression_penalty)
			self.logistic_regression_run(regression_penalty, is_binary_classification)
		elif model == "tree":
			self.decision_tree_run(is_binary_classification, selected_feature_names, y_names)
		elif model == "svm":
			self.svm_run(is_binary_classification)
		print(" ### classification end ### \n")

	def find_selected_features_names(self, selected_features_indices, selection_method, save_names):
		"""
		Find the names of selected features from the input dataframe

		Parameters
		----------
		selected_features_indices : set
			set of selected features indices
		selection_method : str
			name of applied feature selection method
		save_names : bool
			save the selected feature names in file (True), otherwise do not save

		Returns
		-------
		selected_feature_names : list of str
			selected features name
		"""
		print("Find names of selected features")
		selected_features_names = [feature_name for feature_idx, feature_name in
		                           enumerate(list(self.X_train_df.columns)) if feature_idx in selected_features_indices]
		print("Selected features:\n{}".format("\n".join(selected_features_names)))

		file_name = selection_method + "_features.txt"
		if save_names:  # write selected features in file
			with open(join(self.out_path, file_name), 'w') as features_file:
				features_file.write("\n".join(selected_features_names))
		print("---")
		return selected_features_names

	def convert_coef2indices(self, coefficients):
		"""
		Convert coefficients to indices
		and return all indices that have coefficients different than 0

		Parameters
		----------
		coefficients : numpy.array
			coefficients of cofactors computed by optimizing logistic regression with Lasso penalty

		Returns
		-------
		selected_features : set
			selected features from logistic with Lasso
		"""
		# print("For logistic regression with Lasso penalty, get selected features")
		selected_indices = set()
		for idx, coef in enumerate(coefficients.T):
			if coef.all() != 0.0:
				selected_indices.add(idx)
		return selected_indices

	def svm_run(self, is_binary_classification):
		"""
		Run SVM classifier to predict diagnosis label

		Parameters
		----------
		is_binary_classification : bool
			this is a binary classification (True), otherwise it is multi-class (False)

		Returns
		-------
		None
		"""
		print("Run SVM classifier")
		if self.X_test is None:
			print("Run stratified cross validation")
			# in cross validation evaluation,
			# all training data are used for both training and testing
			# after being separated during each split
			X, y = self.X_train, self.y_train
			skf = StratifiedKFold(n_splits=5, random_state=self.random_state, shuffle=True)
			fold_idx = 0
			perf_fold = {'accuracy': [], 'roc-auc': []}

			for train_index, test_index in iter(skf.split(X, y)):
				fold_idx += 1
				X_train, y_train = X[train_index], y[train_index]
				X_test, y_test = X[test_index], y[test_index]
				if is_binary_classification:
					estimator = SVC(kernel='linear', random_state=self.random_state)
				else:  # for multi-class enable to output probability over classes
					estimator = SVC(kernel='linear', random_state=self.random_state, probability=True)
				estimator.fit(X_train, y_train)
				y_test_pred = estimator.predict(X_test)

				# calculate performance for current fold train/test
				perf_fold['accuracy'].append(np.mean(y_test_pred.ravel() == y_test.ravel()) * 100)
				if is_binary_classification:
					perf_fold['roc-auc'].append(roc_auc_score(y_test, y_test_pred))
				else:
					y_test_pred_prob = estimator.predict_proba(X_test)
					num_classes = np.unique(y_test).shape[0]
					label_indices = np.arange(num_classes)
					perf_fold['roc-auc'].append(
						roc_auc_score(y_test, y_test_pred_prob, average='macro', multi_class='ovo',
						              labels=label_indices))
			print(" === Performance over folds === ")
			print("max acc= {}, min acc= {}".format(max(perf_fold['accuracy']), min(perf_fold['accuracy'])))
			print("mean acc= {}".format(sum(perf_fold['accuracy']) / len(perf_fold['accuracy'])))
			print("max roc-auc= {}, min roc-auc= {}".format(max(perf_fold['roc-auc']), min(perf_fold['roc-auc'])))
			print("mean roc-auc= {}".format(sum(perf_fold['roc-auc']) / len(perf_fold['roc-auc'])))

	def decision_tree_run(self, is_binary_classification, selected_feature_names, y_names):
		"""
		Run decision tree to classify diagnosis label

		Parameters
		----------
		is_binary_classification : bool
			this is a binary classification (True), otherwise it is multi-class (False)
		selected_feature_names : list of str
			list of selected feature names
		y_names : list of str
			list of target variable y, categorical names
		Returns
		-------
		None
		"""
		print("Run decision tree classifier")
		if self.X_test is None:
			print("Run stratified cross validation")
			# in cross validation evaluation,
			# all training data are used for both training and testing
			# after being separated during each split
			X, y = self.X_train, self.y_train
			skf = StratifiedKFold(n_splits=5, random_state=self.random_state, shuffle=True)
			fold_idx = 0
			perf_fold = {'accuracy': [], 'roc-auc': []}

			# for each fold: train on k-1 folds and test on the last one
			# then aggregate the average performance over folds
			for train_index, test_index in iter(skf.split(X, y)):
				fold_idx += 1
				X_train, y_train = X[train_index], y[train_index]
				X_test, y_test = X[test_index], y[test_index]
				estimator = DecisionTreeClassifier()
				estimator.fit(X_train, y_train)
				y_test_pred = estimator.predict(X_test)

				# calculate performance for current fold train/test
				perf_fold['accuracy'].append(np.mean(y_test_pred.ravel() == y_test.ravel()) * 100)
				if is_binary_classification:
					perf_fold['roc-auc'].append(roc_auc_score(y_test, y_test_pred))
				else:
					y_test_pred_prob = estimator.predict_proba(X_test)
					num_classes = np.unique(y_test).shape[0]
					label_indices = np.arange(num_classes)
					perf_fold['roc-auc'].append(
						roc_auc_score(y_test, y_test_pred_prob, average='macro', multi_class='ovo',
						              labels=label_indices))
				# visualize the classification tree
				print("Plot decision tree for fold index: {}".format(fold_idx))
				tree_features = selected_feature_names
				tree_y = y_names
				fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi=600)
				tree.plot_tree(estimator, feature_names=tree_features, class_names=tree_y, filled=True)
				fig.savefig(join(self.out_path, 'tree_fold' + str(fold_idx) + '.png'))
			print(" === Performance over folds === ")
			print("max acc= {}, min acc= {}".format(max(perf_fold['accuracy']), min(perf_fold['accuracy'])))
			print("mean acc= {}".format(sum(perf_fold['accuracy']) / len(perf_fold['accuracy'])))
			print("max roc-auc= {}, min roc-auc= {}".format(max(perf_fold['roc-auc']), min(perf_fold['roc-auc'])))
			print("mean roc-auc= {}".format(sum(perf_fold['roc-auc']) / len(perf_fold['roc-auc'])))

	def logistic_regression_run(self, optimization_penalty, is_binary_classification):
		"""
		Run logistic regression with specified penalty to classify diagnosis label

		Parameters
		----------
		optimization_penalty : str
			penalty to use in the regression optimization (l2 or l1)

		is_binary_classification : bool
			this is a binary classification (True), otherwise it is multi-class (False)

		Returns
		-------
		None
		"""
		print("Run logistic regression with {} penalty".format(optimization_penalty))
		if self.X_test is None:
			print("Run stratified cross validation")
			if optimization_penalty == "l1":
				regression_l1_features = set()
			# in cross validation evaluation all training data are used for both training and testing
			# after being separated during each split
			X, y = self.X_train, self.y_train

			skf = StratifiedKFold(n_splits=5, random_state=self.random_state, shuffle=True)
			fold_idx = 0
			perf_fold = {'accuracy': [], 'roc-auc': []}

			# for each fold: train on k-1 folds and test on the last one
			# then aggregate the average performance over folds
			for train_index, test_index in iter(skf.split(X, y)):
				fold_idx += 1
				X_train, y_train = X[train_index], y[train_index]
				X_test, y_test = X[test_index], y[test_index]
				estimator = LogisticRegression(solver='liblinear', penalty=optimization_penalty)
				estimator.fit(X_train, y_train)
				y_test_pred = estimator.predict(X_test)

				if optimization_penalty == "l1":  # per fold get the selected features and save their union
					regression_l1_features = regression_l1_features.union(self.convert_coef2indices(estimator.coef_))
				# calcucate performance for current fold train/test
				perf_fold['accuracy'].append(np.mean(y_test_pred.ravel() == y_test.ravel()) * 100)
				if is_binary_classification:
					perf_fold['roc-auc'].append(roc_auc_score(y_test, y_test_pred))
				else:
					y_test_pred_prob = estimator.predict_proba(X_test)
					num_classes = np.unique(y_test).shape[0]
					label_indices = np.arange(num_classes)
					perf_fold['roc-auc'].append(
						roc_auc_score(y_test, y_test_pred_prob, average='macro', multi_class='ovo',
						              labels=label_indices))

			if optimization_penalty == "l1":
				print("number of selected features= {}".format(len(regression_l1_features)))
				regression_l1_features = sorted(regression_l1_features)
				# print("the union of selected features: {}".format(regression_l1_features))
				selection_method = 'logistic_l1'
				self.find_selected_features_names(regression_l1_features, selection_method, save_names=True)

			print(" === Performance over folds === ")
			print("max acc= {}, min acc= {}".format(max(perf_fold['accuracy']), min(perf_fold['accuracy'])))
			print("mean acc= {}".format(sum(perf_fold['accuracy']) / len(perf_fold['accuracy'])))
			print("max roc-auc= {}, min roc-auc= {}".format(max(perf_fold['roc-auc']), min(perf_fold['roc-auc'])))
			print("mean roc-auc= {}".format(sum(perf_fold['roc-auc']) / len(perf_fold['roc-auc'])))
		else:
			print("Learn on train and evaluate on separate test")
			lasso_cv = LassoCV(alphas=None, cv=5, max_iter=100000, normalize=True)
			lasso_cv.fit(self.X_train, self.y_train)
			lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=10000, normalize=True)
			lasso.fit(self.X_train, self.y_train)
			roc_auc = roc_auc_score(self.y_test, lasso.predict(self.X_test))
			print(roc_auc)

	def initialize_model(self, model_name, test_set_exists):
		"""
		Initialize learning model

		Parameters
		----------
		model_name : str
			learning model name
		test_set_exists : bool
			test set exists (True), otherwise all data used for cross validation

		Returns
		-------
		initialized_model : sklearn model
			set up learning model
		"""
		print("Initialize {} model".format(model_name))
		initialized_model = None
		if test_set_exists:
			pass
		else:
			if model_name == "lasso":
				initialized_model = Lasso(max_iter=10000, normalize=True)
		return initialized_model

	def stratified_cross_validate(self, model_name, initialized_model):
		"""
		Apply stratified cross validate to the whole data set

		Parameters
		----------
		model_name : str
			learning model name
		initialized_model : sklearn model
			set up learning model

		Returns
		-------
		None
		"""
		print("Run stratified cross validation for {}".format(model_name))
		skf = StratifiedKFold(n_splits=5, random_state=self.random_state, shuffle=True)

		# in cross validation evaluation all training data are used for both training and testing
		# after being separated during each split
		X, y = self.X_train, self.y_train

		fold_idx = 0
		test_roc_auc_folds = []
		for train_index, test_index in iter(skf.split(X, y)):
			fold_idx += 1
			X_train = X[train_index]
			y_train = y[train_index]
			X_test = X[test_index]
			y_test = y[test_index]
			fold_fit = initialized_model.fit(X_train, y_train)
			if model_name == "lasso":
				## check again!
				lasso = Lasso(alpha=fold_fit.alpha_, max_iter=10000, normalize=True)
				roc_auc = roc_auc_score(y_test, lasso.predict(X_test))
				test_roc_auc_folds.append(roc_auc)

		print(" === {} roc-auc === ".format(model_name))
		for fold_idx, perf_fold in enumerate(test_roc_auc_folds):
			print("fold= {}, roc-auc= {}".format(fold_idx, perf_fold))
		avg_perf = sum(test_roc_auc_folds) / len(test_roc_auc_folds)
		print("Average roc-auc= {}".format(avg_perf))
