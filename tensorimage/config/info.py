import os
import tensorimage.file.reader as tsfr

# User configurations
config_path = os.path.dirname(os.path.abspath(__file__))

file_reader = tsfr.JSONReader("user", os.path.join(config_path, "config.json"))
file_reader.bulk_read()
file_reader.select()
data = file_reader.selected_data

workspace_dir = data["workspace_dir"]
tensorimage_path = data["tensorimage_path"]

predictions_base_filename = data["predictions_base_filename"]

# Program configurations
training_metafile_path = workspace_dir+'metadata/training/meta.json'
classification_metafile_path = workspace_dir+'metadata/classification/meta.json'
dataset_metafile_path = workspace_dir+'metadata/data/meta.json'
id_management_file_path = workspace_dir+'metadata/id.json'

base_training_data_store_path = workspace_dir+'user/training_datasets/'
base_unclassified_data_store_path = workspace_dir+'user/unclassified_datasets/'
base_predictions_store_path = workspace_dir+'user/predictions/'
base_trained_models_store_path = workspace_dir+'user/trained_models/'
