from nnir.src.config import external_working_directory_path
from nnir.src.man.mkdir import mkdir
from nnir.src.man.mkemptfile import mkemptfile
from nnir.src.man.writer import JSONWriter

dir_paths = [external_working_directory_path,
             external_working_directory_path + 'user',
             external_working_directory_path + 'user/datasets',
             external_working_directory_path + 'user/trained_models',
             external_working_directory_path + 'user/data',
             external_working_directory_path + 'user/training_images',
             external_working_directory_path + 'user/unclassified_images',
             external_working_directory_path + 'user/layer_activations',
             external_working_directory_path + 'metadata',
             external_working_directory_path + 'metadata/classification',
             external_working_directory_path + 'metadata/data',
             external_working_directory_path + 'metadata/training']

file_paths = [external_working_directory_path + 'metadata/id.json',
              external_working_directory_path + 'metadata/classification/meta.json',
              external_working_directory_path + 'metadata/data/meta.json',
              external_working_directory_path + 'metadata/training/meta.json',
              external_working_directory_path + 'metadata/training/accuracy.json']


for dir_path in dir_paths:
    mkdir(dir_path)

__version__ = 'v1.0.0'
