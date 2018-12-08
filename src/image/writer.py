from numpy import unique
from progress import bar

from tensorimage.src.file.writer import *
from tensorimage.src.file.reader import *
from tensorimage.src.config import *
from tensorimage.src.os.mkdir import mkdir


class DataWriter:
    def __init__(self,
                 image_data,
                 data_name,
                 dataset_name,
                 img_dims):
        self.image_data = image_data
        self.data_name = data_name
        self.dataset_name = dataset_name
        self.metadata_writer = JSONWriter(self.data_name, dataset_metafile_path)
        self.img_dims = img_dims

        self.data_dir = base_unclassified_data_store_path+self.dataset_name+'/'
        self.csv_writer = CSVWriter(self.data_dir+'data.csv')

        mkdir(self.data_dir)

        self.n_classes = 0

    def write_image_data(self):
        writing_progress = bar.Bar("Writing images: ", max=len(self.image_data))
        for image_data_inst in self.image_data:
            self.csv_writer.write(image_data_inst)
            writing_progress.next()
        self._write_metadata()

    def _write_metadata(self):
        self.metadata_writer.update(
            dataset_type="unclassified",
            n_columns=str(len(self.image_data[0])),
            data_path=self.data_dir + 'data.csv',
            n_classes=self.n_classes,
            width=self.img_dims[0],
            height=self.img_dims[1],
            dataset_name=self.dataset_name)
        self.metadata_writer.write()


class TrainingDataWriter:
    def __init__(self,
                 image_data,
                 data_name,
                 dataset_name,
                 img_dims):
        """
        :param image_data: image data to be written to CSV file
        :param data_name: unique name used to identify extracted image data
        :param dataset_name: image dataset name from which to read images
        :param img_dims: size for dataset images (all sizes should be the same)
        """
        # Store parameters
        self.image_data = image_data
        self.data_name = data_name
        self.dataset_name = dataset_name
        self.img_dims = img_dims
        self.metadata_writer = JSONWriter(self.data_name, dataset_metafile_path)

        # Read image labels text file and store it
        self.labels_path = base_training_data_store_path+dataset_name+'/labels.txt'
        self.path_reader = TXTReader(self.labels_path)
        self.path_reader.read_raw()
        self.path_reader.parse()
        self.labels = self.path_reader.parsed_data

        # Store path where to save output image data with labels and define CSVWriter object
        self.data_dir = workspace_dir+'user/training_datasets/'+self.dataset_name+'/'
        self.csv_writer = CSVWriter(self.data_dir+'data.csv')

        # Create storing folder for output image data
        mkdir(self.data_dir)

        # Store number of classes
        self.n_classes = len(unique(self.labels))

    def write_image_data(self):
        writing_progress = bar.Bar("Writing images: ", max=len(self.image_data))
        for imn in range(len(self.image_data)):
            self.image_data[imn].append(self.labels[imn])
            self.csv_writer.write(self.image_data[imn])
            writing_progress.next()
        self._write_metadata()

    def _write_metadata(self):
        # Update necessary parameters for training process as metadata, to reduce required user input
        self.metadata_writer.update(
            dataset_type="training",
            n_columns=len(self.image_data[0]) + 1,
            data_path=self.data_dir + 'data.csv',
            n_classes=self.n_classes,
            data_len=len(self.image_data[0]),
            width=self.img_dims[0],
            height=self.img_dims[1],
            dataset_name=self.dataset_name)
        self.metadata_writer.write()

