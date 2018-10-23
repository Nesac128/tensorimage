from tensorimage.src.config import workspace_dir
from tensorimage.src.man.mkdir import mkdir
from tensorimage.src.man.mkemptfile import mkemptfile

dir_paths = [workspace_dir,
             workspace_dir + 'user',
             workspace_dir + 'user/datasets',
             workspace_dir + 'user/trained_models',
             workspace_dir + 'user/data',
             workspace_dir + 'user/training_images',
             workspace_dir + 'user/unclassified_images',
             workspace_dir + 'user/layer_activations',
             workspace_dir + 'metadata',
             workspace_dir + 'metadata/classification',
             workspace_dir + 'metadata/data',
             workspace_dir + 'metadata/training']

file_paths = [workspace_dir + 'metadata/id.json',
              workspace_dir + 'metadata/classification/meta.json',
              workspace_dir + 'metadata/data/meta.json',
              workspace_dir + 'metadata/training/meta.json',
              workspace_dir + 'metadata/training/accuracy.json']


for dir_path in dir_paths:
    mkdir(dir_path)

for file_path in file_paths:
    mkemptfile(file_path)

__version__ = 'v1.0.0'
