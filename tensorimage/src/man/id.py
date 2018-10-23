from src.config import *
from src.man.reader import *
from src.man.writer import *


class ID:
    def __init__(self, opt):
        self.opt = opt

        self.idfile_id = None
        self.select_idfile_id()

        self.json_reader = JSONReader(self.idfile_id, id_management_file_path)
        self.json_writer = JSONWriter(self.idfile_id, id_management_file_path)

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
