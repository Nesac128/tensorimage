import pandas as pd
import json

from src.config import *


class JSONReader:
    def __init__(self, id: int, file_path):
        self.id = id
        self.file_path = file_path

        self.bulk_data = None
        self.selected_data = None

    def bulk_read(self):
        with open(self.file_path, 'r') as jsonfile:
            self.bulk_data = json.load(jsonfile)
            jsonfile.close()

    def select(self):
        self.selected_data = self.bulk_data[self.id]


class CSVReader:
    def __init__(self, file_path):
        self.file_path = file_path

        self.X = None
        self.Y = None

    def read_training_dataset(self, m1, m2):
        df = pd.read_csv(self.file_path, header=None)
        X = df[df.columns[0:m1]].values
        Y = df[df.columns[m1:m2]]

        self.X = X
        self.Y = Y

    def read_file(self):
        df = pd.read_csv(self.file_path, header=None)
        X = df[df.columns].values

        self.X = X


class TXTReader:
    def __init__(self, file_path):
        self.file_path = file_path

        self.data = None
        self.parsed_data = []

    def read_raw(self):
        with open(self.file_path, 'r') as txtfile:
            self.data = txtfile.read()

    def parse(self):
        self.parsed_data = self.data.split('\n')
        del self.parsed_data[-1]
