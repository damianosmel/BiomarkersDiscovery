from os import makedirs

def create_dir(base_path, type_name):
	"""
	Create directory based on the base path

	Parameters
	----------
	base_path : str
		path for directory
	type_name : str
		folder type name (e.g output)

	Returns
	-------
	None
	"""
	print("Creating {} folder:".format(type_name))
	print("Folder does not exist => create path: {}".format(base_path))
	makedirs(base_path, exist_ok=True)
