import csv
import json
from tensorimage.file.reader import JSONReader


class JSONWriter:
    def __init__(self, key, file_path):
        """
        :param key: key to write data to
        :param file_path: file path to write/update data to
        """
        self.key = key
        self.file_path = file_path
        self.udata = None

        self.json_reader = JSONReader(self.key, self.file_path)

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
                fdata[self.key][key] = val
        except KeyError:
            fdata[self.key] = {}
            for key, val in zip(keys, values):
                fdata[self.key][key] = val
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
