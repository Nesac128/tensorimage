import importlib
import importlib.util

# from src.config import *
# from src.man.reader import *
# from src.man.writer import *


def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reader = module_from_file('reader', '/home/planetgazer8360/PycharmProjects/nnir/nnir/src/man/reader.py')
writer = module_from_file('writer', '/home/planetgazer8360/PycharmProjects/nnir/nnir/src/man/writer.py')
config = module_from_file('config', '/home/planetgazer8360/PycharmProjects/nnir/nnir/src/config.py')


class ID:
    def __init__(self, opt):
        self.opt = opt

        self.idfile_id = None
        self.select_idfile_id()

        self.json_reader = reader.JSONReader(self.idfile_id, config.id_management_file_path)
        self.json_writer = writer.JSONWriter(self.idfile_id, config.id_management_file_path)

        self.id = None

    def select_idfile_id(self):
        if self.opt == "sess":
            self.idfile_id = '0'
        elif self.opt == "dataset":
            self.idfile_id = '1'
        elif self.opt == "training":
            self.idfile_id = '2'
        elif self.opt == "classification":
            self.idfile_id = '3'
        elif self.opt == "accuracy":
            self.idfile_id = '4'

    def add(self):
        self.read()
        self.json_writer.update(id=int(self.id)+1)
        self.json_writer.write()

    def read(self):
        self.json_reader.bulk_read()
        self.json_reader.select()
        self.id = str(self.json_reader.selected_data['id'])
