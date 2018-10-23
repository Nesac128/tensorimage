from numpy import unique
from progress import bar

from src.man.writer import *
from src.man.reader import *
from src.man.id import ID
from src.config import *
from src.man.mkdir import mkdir


class DataWriter:
    def __init__(self,
                 data,
                 filename,
                 dataset_name,
                 img_dims,
                 metadata_writer,
                 id_name):
        self.data = data
        self.filename = filename
        self.dataset_name = dataset_name
        self.metadata_writer = metadata_writer
        self.img_dims = img_dims
        self.id_name = id_name

        self.data_dir = base_unclassified_data_store_path+self.dataset_name
        self.csv_writer = CSVWriter(self.data_dir+'/'+self.filename)

        mkdir(self.data_dir)

        self.n_classes = 0

        self.id_man = ID('dataset')
        self.id_man.read()
        self.nid = int(self.id_man.id)

        self.nid_names_writer = JSONWriter(self.id_name, nid_names_metafile_path)

    def write_image_data(self):
        writing_progress = bar.Bar("Writing images: ", max=len(self.data))
        for image_data_inst in self.data:
            self.csv_writer.write(image_data_inst)
            writing_progress.next()

    def write_metadata(self):
        self.metadata_writer.update(
            n_columns=str(len(self.data[0])),
            data_path=workspace_dir + 'user/data/unclassified/' + self.dataset_name + '/' + self.filename,
            n_classes=self.n_classes,
            trainable='False',
            width=self.img_dims[0][0],
            height=self.img_dims[0][1],
            name=self.id_name,
            dataset_name=self.dataset_name)
        self.metadata_writer.write()

        self.nid_names_writer.update(id=self.nid+1)
        self.nid_names_writer.write()


class TrainingDataWriter:
    def __init__(self,
                 data,
                 filename,
                 dataset_name,
                 img_dims,
                 metadata_writer,
                 id_name):
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
        self.id_name = id_name

        self.nid_names_writer = JSONWriter(self.id_name, nid_names_metafile_path)

        # Read image labels text file and store it
        self.labels_path = workspace_dir+'user/datasets/'+dataset_name+'/labels.txt'
        self.PathReader = TXTReader(self.labels_path)
        self.PathReader.read_raw()
        self.PathReader.parse()
        self.labels = self.PathReader.parsed_data

        # Read current data ID from id.json and store it
        self.id_man = ID('dataset')
        self.id_man.read()
        self.nid = int(self.id_man.id)

        # Store path where to save output image data with labels and define CSVWriter object
        self.data_dir = workspace_dir+'user/data/training/'+self.dataset_name+'/'
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
            n_columns=len(self.data[0]) + 1,
            data_path=self.data_dir + self.filename,
            n_classes=self.n_classes,
            trainable='True',
            type='Image',
            data_len=len(self.data[0]),
            width=self.img_dims[0][0],
            height=self.img_dims[0][1],
            name=self.id_name)
        self.metadata_writer.write()

        self.nid_names_writer.update(id=self.nid+1)
        self.nid_names_writer.write()
