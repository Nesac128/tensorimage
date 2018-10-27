from numpy import unique
from progress import bar

from src.man.writer import *
from src.man.reader import *
from src.config import *
from src.man.mkdir import mkdir


class DataWriter:
    def __init__(self,
                 data,
                 filename,
                 dataset_name,
                 img_size,
                 metadata_writer):
        self.data = data
        self.filename = filename
        self.dataset_name = dataset_name
        self.metadata_writer = metadata_writer
        self.img_dims = img_size

        self.data_dir = base_unclassified_data_store_path+self.dataset_name
        self.csv_writer = CSVWriter(self.data_dir+'/'+self.filename)

        mkdir(self.data_dir)

        self.n_classes = 0

    def write_image_data(self):
        writing_progress = bar.Bar("Writing images: ", max=len(self.data))
        for image_data_inst in self.data:
            self.csv_writer.write(image_data_inst)
            writing_progress.next()

    def write_metadata(self):
        self.metadata_writer.update(
            dataset_type="unclassified",
            n_columns=str(len(self.data[0])),
            data_path=workspace_dir + 'user/unclassified_datasets/' + self.dataset_name + '/' + self.filename,
            n_classes=self.n_classes,
            width=self.img_dims[0][0],
            height=self.img_dims[0][1],
            dataset_name=self.dataset_name)
        self.metadata_writer.write()


class TrainingDataWriter:
    def __init__(self,
                 data,
                 filename,
                 dataset_name,
                 img_dims,
                 metadata_writer):
        """
        :param data: data to be written to CSV file
        :param filename: CSV filename
        :param dataset_name: image dataset name from which to read images
        :param img_dims: size for dataset images (all sizes should be the same)
        :param metadata_writer: metadata_writer class used in loader.py and passed to writer.py
        :param id_name: unique name to identify extracted data
        """
        # Store parameters
        self.data = data
        self.filename = filename
        self.dataset_name = dataset_name
        self.img_dims = img_dims
        self.metadata_writer = metadata_writer

        # Read image labels text file and store it
        self.labels_path = base_training_data_store_path+dataset_name+'/labels.txt'
        self.path_reader = TXTReader(self.labels_path)
        self.path_reader.read_raw()
        self.path_reader.parse()
        self.labels = self.path_reader.parsed_data

        # Store path where to save output image data with labels and define CSVWriter object
        self.data_dir = workspace_dir+'user/training_datasets/'+self.dataset_name+'/'
        self.csv_writer = CSVWriter(self.data_dir+self.filename)

        # Create storing folder for output image data
        mkdir(self.data_dir)

        # Store number of classes
        self.n_classes = len(unique(self.labels))

    def write_image_data(self):
        writing_progress = bar.Bar("Writing images: ", max=len(self.data))
        for imn in range(len(self.data)):
            self.data[imn].append(self.labels[imn])
            self.csv_writer.write(self.data[imn])
            writing_progress.next()

    def write_metadata(self):
        # Update necessary parameters for training process as metadata, to reduce required user input
        self.metadata_writer.update(
            dataset_type="training",
            n_columns=len(self.data[0]) + 1,
            data_path=self.data_dir + self.filename,
            n_classes=self.n_classes,
            data_len=len(self.data[0]),
            width=self.img_dims[0][0],
            height=self.img_dims[0][1],
            dataset_name=self.dataset_name)
        self.metadata_writer.write()

