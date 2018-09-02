import pandas as pd
from shutil import copyfile
from numpy import unique

from nnir.pcontrol import *


class RawTrainLoader:
    def __init__(self, path_file):
        self.sess = Sess()

        self.pfile = path_file
        assert os.path.exists(path_file), path_file + " does not exist"

        self.reader = Reader(path_file)

        self.file_paths = self.reader.clean_read()

    def metaman(self, n):
        sess_id = self.sess.read()
        meta = MetaData(int(sess_id))
        df = pd.read_csv(self.file_paths[n], header=None)

        meta.write(path_file=self.pfile)
        n_columns = str(len(df.columns))

        meta.write(n_columns=n_columns)
        labels = list(df[df.columns[-1]])
        meta.write(data_path=external_working_directory_path+'data/training/'+self.file_paths[n].split('/')[-1])
        meta.write(n_classes=str(len(list(unique(labels)))))
        meta.write(trainable='True')
        meta.write(type='Raw')

    def main(self):
        for n, path in enumerate(self.file_paths):
            sess_id = self.sess.read()

            if not os.path.exists(nnir_path+'meta/sess/'+sess_id):
                os.mkdir(nnir_path+'meta/sess/'+sess_id)

            self.metaman(n)

            if not os.path.exists(external_working_directory_path+'data/training/'+self.file_paths[n].split('/')[-1]):
                copyfile(path, external_working_directory_path+'data/training/'+self.file_paths[n].split('/')[-1])

            pman = PathManager()
            pman.cpaths()

            self.sess.add()


class RawLoader:
    def __init__(self, path_file):
        self.sess = Sess()

        self.pfile = path_file
        assert os.path.exists(path_file), path_file + " does not exist"

        self.reader = Reader(path_file)

        self.file_paths = self.reader.clean_read()

    def metaman(self, n):
        sess_id = self.sess.read()
        meta = MetaData(int(sess_id))
        df = pd.read_csv(self.file_paths[n], header=None)

        meta.write(path_file=self.pfile)
        n_columns = str(len(df.columns))

        meta.write(n_columns=n_columns)
        meta.write(data_path=external_working_directory_path + 'data/training/' + self.file_paths[n].split('/')[-1])
        meta.write(n_classes=str(0))
        meta.write(trainable='False')
        meta.write(type='Raw')

    def main(self):
        for n, path in enumerate(self.file_paths):
            sess_id = self.sess.read()

            if not os.path.exists(nnir_path+'meta/sess/'+sess_id):
                os.mkdir(nnir_path+'meta/sess/'+sess_id)

            self.metaman(n)

            if not os.path.exists(external_working_directory_path+'data/training/'+self.file_paths[n].split('/')[-1]):
                copyfile(path, external_working_directory_path+'data/training/'+self.file_paths[n].split('/')[-1])

            pman = PathManager()
            pman.cpaths()

            self.sess.add()


rtl = RawLoader('/home/planetgazer8360/Desktop/test.txt')
rtl.main()
