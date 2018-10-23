import csv
import json
from src.man.reader import JSONReader


class JSONWriter:
    def __init__(self, id, file_path):
        """
        :param id: id (key) to write data to
        :param file_path: file path to write/update data to
        """
        self.id = id
        self.file_path = file_path
        self.udata = None

        self.json_reader = JSONReader(self.id, self.file_path)

    def parse(self):
        keys = [key for key in self.udata]
        values = [val for val in self.udata.values()]
        return keys, values

    def read(self):
        self.json_reader.bulk_read()
        return self.json_reader.bulk_data

    def update(self, **udata):
        """
        :param udata: **kwargs data to write in JSON file
        """
        self.udata = udata

        keys, values = self.parse()

        fdata = self.read()
        try:
            for key, val in zip(keys, values):
                fdata[self.id][key] = val
        except KeyError:
            fdata[self.id] = {}
            for key, val in zip(keys, values):
                fdata[self.id][key] = val
        self.udata = fdata

    def write(self):
        with open(self.file_path, 'w') as jsonfile:
            json.dump(self.udata, jsonfile, indent=3)


class CSVWriter:
    def __init__(self, file_path):
        self.file_path = file_path

    def write(self, data):
        with open(self.file_path, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(data)


class TXTWriter:
    def __init__(self, file_path):
        self.file_path = file_path

    def write(self, *data):
        with open(self.file_path, 'a') as txtfile:
            for inst in data:
                txtfile.write(inst+'\n')












