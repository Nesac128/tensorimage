import warnings
import csv
import os
from numpy import unique
from progress import bar

from src.pcontrol import *


class DataWriter:
    def __init__(self, data, filename, dataset_name, imsize):
        self.raw_data = data
        self.filename = filename
        self.dataset_name = dataset_name
        self.Meta = MetaData()
        self.imsize = imsize

        if not os.path.exists(external_working_directory_path+'data/unclassified/'+self.dataset_name):
            os.mkdir(external_working_directory_path+'data/unclassified/'+self.dataset_name)

    def main(self):
        for image_data in self.raw_data:
            self.writeCSV(image_data)
        self.Meta.write(Sess().read(), n_columns=str(len(self.raw_data[0])))
        self.Meta.write(Sess().read(), data_path=external_working_directory_path+'data/unclassified/' +
                        self.dataset_name+'/'+self.filename)
        self.Meta.write(Sess().read(), n_classes='0')
        self.Meta.write(Sess().read(), trainable='False')
        self.Meta.write(Sess().read(), width=self.imsize[0])
        self.Meta.write(Sess().read(), height=self.imsize[1])

    def writeCSV(self, img):
        with open(external_working_directory_path+'data/unclassified/'+self.dataset_name+'/' +
                  self.filename, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(img)


class TrainingDataWriter:
    def __init__(self, data, filename, dataset_name, imsize):
        self.input_data = data
        self.filename = filename
        self.dataset_name = dataset_name
        self.imsize = imsize

        self.labels_path = external_working_directory_path+'datasets/'+dataset_name+'/labels.txt'

        self.Meta = MetaData()
        self.Reader = Reader(self.labels_path)

        self.labels = self.Reader.clean_read()

        self.wsid = Sess().read()

        if not os.path.exists(external_working_directory_path+'data/training/'+self.dataset_name):
            os.mkdir(external_working_directory_path+'data/training/'+self.dataset_name)

    def n_classes(self):
        return str(len(unique(self.labels)))

    def verify_dimensions(self):
        file_reader = Reader(external_working_directory_path+'data/training/'+self.dataset_name+'/'+self.filename)
        data = file_reader.clean_read()
        hist = []
        for image in data:
            hist.append(len(image))
        for n in range(len(data)):
            for pn in range(n):
                if hist[n] != hist[pn]:
                    warnings.warn("Dimensions of data for every image do not match overall. \
                                                  This will cause an error when trying to train the data")
                    break

    def main(self):
        self.metaWriter()
        writing_progress = bar.Bar("Writing images: ", max=len(self.input_data))
        for imn in range(len(self.input_data)):
            self.input_data[imn].append(self.labels[imn])
            self.writeCSV(self.input_data[imn])
            writing_progress.next()
        print("")

    def writeCSV(self, img):
        with open(external_working_directory_path+'data/training/'+self.dataset_name+'/'+self.filename, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(img)

    def metaWriter(self):
        self.Meta.write(self.wsid, n_columns=str(len(self.input_data[0])+1))
        self.Meta.write(self.wsid, data_path=external_working_directory_path+'data/training/'+self.dataset_name+'/' +
                        self.filename)
        self.Meta.write(self.wsid, n_classes=self.n_classes())
        self.Meta.write(self.wsid, trainable='True')
        self.Meta.write(self.wsid, type='Image')
        self.Meta.write(self.wsid, data_len=str(len(self.input_data[0])))
        self.Meta.write(self.wsid, width=self.imsize[0])
        self.Meta.write(self.wsid, height=self.imsize[1])
