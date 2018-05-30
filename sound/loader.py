import sox
import re

import nnir.pcontrol as np
from config import *


class Loader:
    def __init__(self, wav_path_file):
        self.path_file = wav_path_file
        self.wav_path_reader = np.Reader(wav_path_file)

        self.sess = np.Sess()

        self.files = []

        for fpath in self.wav_path_reader.clean_read():
            self.files.append(fpath)

        self.Meta = np.MetaData(self.sess.read())

        self.sess.ndir()

    def meta_writer(self, data):
        self.Meta.write(path_file=self.path_file)
        self.Meta.write(n_columns=str(len(data[0])))

    def wav_to_dat(self):
        transformer = sox.Transformer()
        for wavpath in self.files:
            transformer.build(wavpath, nnir_path + 'tmp/' +
                              wavpath.split('/')[-1].split('.')[0]+'.dat')

    def clean_dat(self):
        for wavpath in self.files:
            with open(nnir_path+'tmp/'+wavpath.split('/')[-1].split('.')[0]+'.dat', 'r') as datfile:
                read_lines = datfile.readlines()
                lines = []
                for n, line in enumerate(read_lines):
                    if n <= 1:
                        continue
                    ln = line.strip('|').strip()
                    sentence = re.sub(r"\s+", ",", ln, flags=re.UNICODE)
                    ln = float(sentence.split(',')[0]), float(sentence.split(',')[1])
                    lines.append(ln)
            del lines[0], lines[0]
            os.remove(nnir_path+'tmp/'+wavpath.split('/')[-1].split('.')[0]+'.dat')
            yield lines

    def main(self):
        self.wav_to_dat()
        gdata = self.clean_dat()
        dtgl = []
        for dtg in gdata:
            dtgl.append(dtg)
        self.meta_writer(dtgl)
        pman = np.PathManager()
        pman.cpaths()

        return gdata
