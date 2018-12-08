import tensorimage.src.file.reader as tsfr
import os

file_reader = tsfr.JSONReader("user", os.getcwd()+'/config.json')
file_reader.bulk_read()
file_reader.select()
data = file_reader.selected_data

# User configurations
workspace_dir = data["workspace_dir"]
tensorimage_path = data["tensorimage_path"]

predictions_base_filename = data["predictions_base_filename"]

# Program configurations
training_metafile_path = workspace_dir+'metadata/training/meta.json'
classification_metafile_path = workspace_dir+'metadata/classification/meta.json'
dataset_metafile_path = workspace_dir+'metadata/data/meta.json'
id_management_file_path = workspace_dir+'metadata/id.json'
nid_names_metafile_path = workspace_dir+'metadata/nid_names.json'

base_training_data_store_path = workspace_dir+'user/training_datasets/'
base_unclassified_data_store_path = workspace_dir+'user/unclassified_datasets/'
base_predictions_store_path = workspace_dir+'user/predictions/'
base_trained_models_store_path = workspace_dir+'user/trained_models/'
