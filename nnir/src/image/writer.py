import csv
import os
from numpy import unique
from progress import bar

from src.pcontrol import *
from src.man.writer import *
from src.man.reader import *
from src.meta.id import ID
from src.config import *
from src.man.mkdir import mkdir


class DataWriter:
    def __init__(self, data, filename, dataset_name, img_dims, metadata_writer):
        self.data = data
        self.filename = filename
        self.dataset_name = dataset_name
        self.metadata_writer = metadata_writer
        self.img_dims = img_dims

        self.data_dir = base_unclassified_data_store_path+self.dataset_name
        self.csv_writer = CSVWriter(self.data_dir)

        mkdir(self.data_dir)

        self.n_classes = 0

    def main(self):
        for image_data_inst in self.data:
            self.csv_writer.write(image_data_inst)
        self.metadata_writer.update(
            n_columns=str(len(self.data[0])),
            data_path=external_working_directory_path + 'data/unclassified/' + self.dataset_name + '/' + self.filename,
            n_classes=self.n_classes,
            trainable='False',
            width=self.img_dims[0][0],
            height=self.img_dims[0][1]
        )
        self.metadata_writer.write()


class TrainingDataWriter:
    def __init__(self, data, filename, dataset_name, img_dims, metadata_writer):
        """
        :param data: data to be written to CSV file
        :param filename: CSV filename
        :param dataset_name: image dataset name from which to read images
        :param imsize: size for dataset images (all sizes should be the same)
        :param metadata_writer: metadata_writer class used in loader.py and passed to writer.py
        """
        # Store parameters
        self.input_data = data
        self.filename = filename
        self.dataset_name = dataset_name
        self.img_dims = img_dims
        self.MetaWriter = metadata_writer

        # Read image labels text file and store it
        self.labels_path = external_working_directory_path+'datasets/'+dataset_name+'/labels.txt'
        self.PathReader = TXTReader(self.labels_path)
        self.PathReader.read_raw()
        self.PathReader.parse()
        self.labels = self.PathReader.parsed_data

        # Read current data ID from id.json and store it
        self.id_man = ID('dataset')
        self.id_man.read()
        self.wdid = self.id_man.id

        # Store path where to save output image data with labels and define CSVWriter object
        self.data_dir = external_working_directory_path+'data/training/'+self.dataset_name+'/'
        self.csv_writer = CSVWriter(self.data_dir+self.filename)

        # Create storing folder for output image data
        mkdir(self.data_dir)

        # Store number of classes
        self.n_classes = len(unique(self.labels))

    def join_data_labels(self):
        writing_progress = bar.Bar("Writing images: ", max=len(self.input_data))
        for imn in range(len(self.input_data)):
            self.input_data[imn].append(self.labels[imn])
            self.csv_writer.write(self.input_data[imn])
            writing_progress.next()

    def write_metadata(self):
        # Update necessary parameters for training process as metadata, to reduce required user input
        self.MetaWriter.update(
            n_columns=len(self.input_data[0]) + 1,
            data_path=self.data_dir + self.filename,
            n_classes=self.n_classes,
            trainable='True',
            type='Image',
            data_len=len(self.input_data[0]),
            width=self.img_dims[0][0],
            height=self.img_dims[0][1])
        self.MetaWriter.write()
