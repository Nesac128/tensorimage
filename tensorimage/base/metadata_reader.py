import ast

from tensorimage.file.reader import JSONReader
from tensorimage.config.info import training_metafile_path, dataset_metafile_path


class TrainerMetadata:
    def __init__(self, training_name: str):
        self.training_name = training_name

        self.training_metadata_reader = JSONReader(self.training_name, training_metafile_path)
        self.training_metadata_reader.bulk_read()
        self.training_metadata_reader.select()
        training_metadata = self.training_metadata_reader.selected_data
        self.model_name = training_metadata["model_name"]
        self.training_dataset_name = training_metadata["dataset_name"]
        self.n_classes = training_metadata["n_classes"]
        self.architecture = training_metadata["architecture"]
        self.height = training_metadata["height"]
        self.width = training_metadata["width"]
        self.n_channels = training_metadata["n_channels"]


class UnclassifiedDatasetMetadata:
    def __init__(self, data_name: str):
        self.data_name = data_name

        self.metadata_reader = JSONReader(self.data_name, dataset_metafile_path)
        self.metadata_reader.bulk_read()
        self.metadata_reader.select()
        image_metadata = self.metadata_reader.selected_data
        self.data_path = image_metadata["data_path"]
        self.width = image_metadata["width"]
        self.height = image_metadata["height"]
        self.path_file = image_metadata["path_file"]
        self.trainable = image_metadata["trainable"]
        if ast.literal_eval(self.trainable):
            raise AssertionError("Data is trainable")


class TrainingDatasetMetadata:
    def __init__(self, data_name: str):
        self.metadata_reader = JSONReader(data_name, dataset_metafile_path)
        self.metadata_reader.bulk_read()
        self.metadata_reader.select()
        image_metadata = self.metadata_reader.selected_data
        self.n_columns = image_metadata["n_columns"]
        self.training_data_path = image_metadata["data_path"]
        self.n_classes = image_metadata["n_classes"]
        self.dataset_type = image_metadata["dataset_type"]
        if not self.dataset_type == 'training':
            raise AssertionError()
        self.data_len = image_metadata["data_len"]
        self.width = image_metadata["width"]
        self.height = image_metadata["height"]
        self.dataset_name = image_metadata["dataset_name"]
