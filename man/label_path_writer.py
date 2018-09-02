import os
import string
from itertools import product
import json

from config import external_working_directory_path


def write_labels(main_directory_path, dataset_name, label_file_name='labels.txt'):
    bspath = external_working_directory_path + main_directory_path

    folders = os.listdir(bspath)

    keys = []

    for k in product(string.ascii_uppercase, repeat=2):
        keys.append(k[0] + k[1])

    obj_to_lb = {}

    if not os.path.exists(external_working_directory_path + 'datasets/' + dataset_name):
        os.mkdir(external_working_directory_path + 'datasets/' + dataset_name)

    with open(external_working_directory_path + 'datasets/' + dataset_name + '/' + label_file_name, 'a') as labels:
        for i, folder in enumerate(folders):
            for ftp in os.listdir(bspath+folder):
                labels.write(keys[i] + '\n')
            obj_to_lb[folder.split('-')[0]] = keys[i]

    with open(external_working_directory_path + 'datasets/' + dataset_name + '/obj_labels.json', 'w') as txt:
        json.dump(obj_to_lb, txt, indent=3, sort_keys=True)


def write_paths(main_directory_path, dataset_name):
    bspath = external_working_directory_path+main_directory_path

    folders = os.listdir(bspath)

    if not os.path.exists(external_working_directory_path + 'datasets/' + dataset_name):
        os.mkdir(external_working_directory_path + 'datasets/' + dataset_name)

    with open(external_working_directory_path + 'datasets/' + dataset_name + '/paths.txt', 'a') as paths:
        for folder in folders:
            for tp in os.listdir(bspath+'/'+folder):
                paths.write(bspath+folder+'/'+tp+'\n')
