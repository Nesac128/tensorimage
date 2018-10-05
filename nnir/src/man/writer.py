import csv
import json
import importlib
import importlib.util


def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reader = module_from_file('reader', '/home/planetgazer8360/PycharmProjects/nnir/nnir/src/man/reader.py')


class JSONWriter:
    def __init__(self, id: int, file_path):
        """
        :param id: id to write data to (e.g: training_id:14, data_id:98 or sess_id:119)
        :param file_path: file path to write/update data to
        """
        self.id = id
        self.file_path = file_path
        self.udata = None

        self.json_reader = reader.JSONReader(self.id, self.file_path)

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
        :return:
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












