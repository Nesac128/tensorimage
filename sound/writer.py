import csv
from collections import Counter

import nnir.pcontrol as np
from config import external_working_directory_path
from man import sound_iter_clean as sic


class TrainDataWriter:
    def __init__(self, data, fname, labels_path):
        self.data = data
        self.fname = fname

        self.lb_reader = np.Reader(labels_path)
        self.Meta = np.MetaData(np.Sess().read())

        self.labels = self.lb_reader.clean_read()

    def meta_writer(self):
        self.Meta.write(data_path=external_working_directory_path+self.fname+'.csv')
        self.Meta.write(n_classes=len(list(Counter(self.labels).keys())))
        self.Meta.write(trainable='True')
        self.Meta.write(type='Sound')

    def join_data_lb(self):
        for n, lb in enumerate(self.labels):
            try:
                self.data[n].append(lb)
            except IndexError:
                pass

    def write(self):
        with open(external_working_directory_path+self.fname+'.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            for snd in self.data:
                writer.writerow(snd)

    def main(self):
        self.data = sic.snd_itcl(self.data)
        self.join_data_lb()
        self.write()
        self.meta_writer()
        np.Sess().add()


class DataWriter:
    def __init__(self, data, fname):
        self.data = data
        self.fname = fname

        self.Meta = np.MetaData(np.Sess().read())
        self.meta_writer()

    def write(self):
        with open(self.fname+'.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            for snd in self.data:
                writer.writerow(snd)

    def meta_writer(self):
        self.Meta.write(data_path=self.fname+'.csv')
        self.Meta.write(n_classes='0')
        self.Meta.write(trainable='False')
        self.Meta.write(type='Sound')

    def main(self):
        self.write()
        np.Sess().add()
