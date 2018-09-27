import csv
import os
from numpy import unique
from progress import bar

from src.pcontrol import *
from src.man.writer import *
from src.man.reader import *
from src.meta.id import ID
from src.config import *


class DataWriter:
    def __init__(self, data, filename, dataset_name, imsize, metadata_writer):
        self.raw_data = data
        self.filename = filename
        self.dataset_name = dataset_name
        self.metadata_writer = metadata_writer
        self.imsize = imsize

        if not os.path.exists(external_working_directory_path+'data/unclassified/'+self.dataset_name):
            os.mkdir(external_working_directory_path+'data/unclassified/'+self.dataset_name)

    def main(self):
        for image_data in self.raw_data:
            self.writeCSV(image_data)
        self.metadata_writer.update(
            n_columns=str(len(self.raw_data[0])),
            data_path=external_working_directory_path + 'data/unclassified/' + self.dataset_name + '/' + self.filename,
            n_classes='0',
            trainable='False',
            width=self.imsize[0],
            height=self.imsize[1]
        )
        self.metadata_writer.write()

    def writeCSV(self, img):
        with open(external_working_directory_path+'data/unclassified/'+self.dataset_name+'/' +
                  self.filename, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(img)


class TrainingDataWriter:
    def __init__(self, data, filename, dataset_name, imsize, metadata_writer):
        self.input_data = data
        self.filename = filename
        self.dataset_name = dataset_name
        self.imsize = imsize

        self.labels_path = external_working_directory_path+'datasets/'+dataset_name+'/labels.txt'

        self.MetaWriter = metadata_writer

        self.PathReader = TXTReader(self.labels_path)
        self.PathReader.read_raw()
        self.PathReader.parse()
        self.labels = self.PathReader.parsed_data

        self.id_man = ID('dataset')
        self.id_man.read()
        self.wdid = self.id_man.id

        self.data_dir = external_working_directory_path+'data/training/'+self.dataset_name+'/'

        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

    def n_classes(self):
        return len(unique(self.labels))

    def main(self):
        self.write_metadata()
        writing_progress = bar.Bar("Writing images: ", max=len(self.input_data))
        for imn in range(len(self.input_data)):
            self.input_data[imn].append(self.labels[imn])
            self.write_csv(self.input_data[imn])
            writing_progress.next()
        self.id_man.add()

    def write_csv(self, img):
        with open(external_working_directory_path+'data/training/'+self.dataset_name+'/'+self.filename, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(img)

    def write_metadata(self):
        self.MetaWriter.update(
            n_columns=len(self.input_data[0]) + 1,
            data_path=self.data_dir + self.filename,
            n_classes=self.n_classes(),
            trainable='True',
            type='Image',
            data_len=len(self.input_data[0]),
            width=self.imsize[0],
            height=self.imsize[1])
        self.MetaWriter.write()
