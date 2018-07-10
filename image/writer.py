from numpy import unique

from exceptions import *
from nnir.pcontrol import *
from config import external_working_directory_path


class DataWriter:
    def __init__(self, data, fname):
        self.raw_data = data
        self.fname = fname
        self.Meta = MetaData(Sess().read())

    def main(self):
        for image_data in self.raw_data:
            self.writeCSV(image_data)
        print(self.fname)
        self.Meta.write(data_path=os.getcwd()+'/'+self.fname)
        self.Meta.write(n_classes='0')
        self.Meta.write(trainable='False')
        self.Meta.write(type='Image')

    def writeCSV(self, img):
        print(external_working_directory_path)
        with open(external_working_directory_path+self.fname, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(img)


class TrainDataWriter:
    def __init__(self, data, fname, labels_path):
        # Store parameters in variables
        self.input_data = data
        self.fname = fname
        self.labels_path = labels_path

        self.Meta = MetaData(Sess().read())
        self.Reader = Reader(self.labels_path)

        self.labels = self.Reader.clean_read()

    def clabels(self):
        return str(len(unique(self.labels)))

    def ccolumns(self):
        file_reader = Reader(external_working_directory_path+self.fname)
        data = file_reader.clean_read()
        hist = []
        for invn, inv in enumerate(data):
            hist.append(inv)
            if invn == 0:
                continue
            if hist[invn] != [inv for inv in data]:
                raise ColumnNumberError("Number of columns in data file {} not matching".format(self.fname))

    def main(self, **tags):
        print("Began with TrainingDataWriting...")
        print(len(self.input_data), "Input data length")
        for imn in range(len(self.input_data)):
            print(imn, "Image-n")
            self.input_data[imn].append(self.labels[imn])
            self.writeCSV(self.input_data[imn])
        self.metaWriter(tags)
        self.ccolumns()

    def writeCSV(self, img):
        with open(external_working_directory_path+self.fname, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(img)

    def metaWriter(self, tags):
        self.Meta.write(data_path=external_working_directory_path + '/' + self.fname)
        self.Meta.write(n_classes=self.clabels())
        self.Meta.write(trainable='True')
        for tag in list(tags.items()):
            exec('self.Meta.write('+str(tag[0]+'='+str(tag[1])+')'))
