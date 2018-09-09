import csv
import pandas as pd
import os

from src.config import *


class Sess:
    def add(self):
        nid = str(int(self.read())+1)
        with open(nnir_path+'nnir/src/meta/ims.txt', 'w') as sessfile:
            sessfile.write(nid)

    def read(self):
        with open(nnir_path+'nnir/src/meta/ims.txt', 'r') as sessfile:
            id = sessfile.read()
            sessfile.close()
        return str(id)

    def ndir(self):
        if not os.path.exists(nnir_path+'nnir/src/meta/sess/'+str(self.read())):
            os.mkdir(nnir_path+'nnir/src/meta/sess/'+str(self.read()))


class MetaData:
    def __init__(self):
        pass

    def write(self, write_sess_id, **meta):
        with open(nnir_path+'nnir/src/meta/sess/'+write_sess_id+'/meta.txt', 'a') as metadata:
            for key in meta:
                for val in meta.values():
                    print(val)
                    metadata.write(str(key.upper()+'='+str(val)+'\n'))

    def read(self, *tags, sess_id):
        reader = Reader(nnir_path+'nnir/src/meta/sess/' + str(sess_id) + '/meta.txt')
        meta = reader.clean_read()
        for mt in meta:
            for tag in tags:
                if tag.upper() == mt.split('=')[0]:
                    yield mt.split('=')[1]


class Reader:
    def __init__(self, file, delimiter='\n'):
        self.file = file
        self.format = self.file.split('.')[-1]
        self.delm = delimiter

    def read_raw(self):
        if self.format == 'csv':
            df = pd.read_csv(self.file, header=None)
            data = df[df.columns[0:-1]].values

            return data

        elif self.format == 'txt':
            txtfile = open(self.file, 'r')
            return txtfile

    def clean_read(self):
        read = self.read_raw()
        data = []
        for row in read:
            try:
                data.append(row.split('\n')[0])
            except AttributeError:
                data.append([val for val in row])
        return data


class PathManager:
    def __init__(self):
        self.sess = Sess()
        self.crt_sid = self.sess.read()
        self.Meta = MetaData()

        raw_meta = self.Meta.read('path_file', sess_id=self.crt_sid)

        meta = [mt for mt in raw_meta]

        self.pfile = meta[0]

        if not os.path.exists(nnir_path+'nnir/src/meta/sess/' + self.crt_sid + '/impaths.csv'):
            with open(nnir_path+'nnir/src/meta/sess/' + self.crt_sid + '/impaths.csv', 'w') as pathfile:
                pathfile.close()

    def cpaths(self):
        reader = Reader(self.pfile)
        paths = reader.clean_read()
        with open(nnir_path+'nnir/src/meta/sess/' + self.crt_sid + '/impaths.csv', 'a') as pathfile:
            writer = csv.writer(pathfile, delimiter='\n')
            for path in paths:
                writer.writerow([path])
        return True
