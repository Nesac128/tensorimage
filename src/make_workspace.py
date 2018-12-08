from tensorimage.src.config import workspace_dir
from tensorimage.src.os.mkdir import mkdir
from tensorimage.src.os.make_file import make_json_file


def make_workspace():
    dir_paths = [workspace_dir,
                 workspace_dir + 'user',
                 workspace_dir + 'user/trained_models',
                 workspace_dir + 'user/training_datasets',
                 workspace_dir + 'user/unclassified_datasets',
                 workspace_dir + 'user/logs',
                 workspace_dir + 'user/predictions',
                 workspace_dir + 'metadata',
                 workspace_dir + 'metadata/data',
                 workspace_dir + 'metadata/training',
                 workspace_dir + 'metadata/classifying']

    file_paths = [workspace_dir + 'metadata/id.json',
                  workspace_dir + 'metadata/data/meta.json',
                  workspace_dir + 'metadata/training/meta.json']

    for dir_path in dir_paths:
        mkdir(dir_path)

    for file_path in file_paths:
        make_json_file(file_path,  {
            "1": {"id": 0},
            "2": {"id": 0},
            "3": {"id": 0}}) if 'id.' in file_path else make_json_file(file_path, {})
