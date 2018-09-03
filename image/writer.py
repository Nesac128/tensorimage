import warnings
from numpy import unique

from exceptions import *
from nnir.pcontrol import *
from config import external_working_directory_path


class DataWriter:
    def __init__(self, data, fname, dataset_name):
        self.raw_data = data
        self.fname = fname
        self.dataset_name = dataset_name
        self.Meta = MetaData()

        if not os.path.exists(external_working_directory_path+'data/unclassified/'+self.dataset_name):
            os.mkdir(external_working_directory_path+'data/unclassified/'+self.dataset_name)

    def main(self):
        for image_data in self.raw_data:
            self.writeCSV(image_data)
        self.Meta.write(Sess().read(), n_columns=str(len(self.raw_data[0])))
        self.Meta.write(Sess().read(), data_path=external_working_directory_path+'data/unclassified/' +
                        self.dataset_name+'/'+self.fname)
        self.Meta.write(Sess().read(), n_classes='0')
        self.Meta.write(Sess().read(), trainable='False')
        self.Meta.write(Sess().read(), type='Image')

    def writeCSV(self, img):
        with open(external_working_directory_path+'data/unclassified/'+self.dataset_name+'/' +
                  self.fname, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(img)


class TrainingDataWriter:
    def __init__(self, data, file_name, dataset_name):
        self.input_data = data
        self.fname = file_name
        self.dataset_name = dataset_name

        self.labels_path = external_working_directory_path+'datasets/'+dataset_name+'/labels.txt'

        self.Meta = MetaData()
        self.Reader = Reader(self.labels_path)

        self.labels = self.Reader.clean_read()

        self.wsid = Sess().read()

        if not os.path.exists(external_working_directory_path+'data/training/'+self.dataset_name):
            os.mkdir(external_working_directory_path+'data/training/'+self.dataset_name)

    def clbs(self):
        return str(len(unique(self.labels)))

    def verify_dimensions(self):
        file_reader = Reader(external_working_directory_path+'data/training/'+self.dataset_name+'/'+self.fname)
        data = file_reader.clean_read()
        hist = []
        for invn, inv in enumerate(data):
            hist.append(inv)
            if invn == 0:
                continue
            if hist[invn] != [inv for inv in data]:
                warnings.warn("Dimensions of data for every image do not match overall. \
                              This will cause an error when trying to train the data")

    def main(self):
        print("Began with TrainingDataWriting...")
        print(len(self.input_data), "Input data length")
        self.metaWriter()
        for imn in range(len(self.input_data)):
            print(imn, "Image-n")
            self.input_data[imn].append(self.labels[imn])
            self.writeCSV(self.input_data[imn])
        self.verify_dimensions()

    def writeCSV(self, img):
        with open(external_working_directory_path+'data/training/'+self.dataset_name+'/'+self.fname, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(img)

    def metaWriter(self):
        self.Meta.write(self.wsid, n_columns=str(len(self.input_data[0])+1))
        self.Meta.write(self.wsid, data_path=external_working_directory_path+'data/training/'+self.dataset_name+'/' +
                        self.fname)
        self.Meta.write(self.wsid, n_classes=self.clbs())
        self.Meta.write(self.wsid, trainable='True')
        self.Meta.write(self.wsid, type='Image')
        self.Meta.write(self.wsid, data_len=str(len(self.input_data[0])-1))
