import os
import string
from itertools import product
import json
from progress import bar

from src.config import external_working_directory_path


def write_labels(main_directory_path, dataset_name):
    bspath = external_working_directory_path + 'user/' + main_directory_path + '/'

    folders = os.listdir(bspath)
    n_images = 0
    for folder in folders:
        n_images += len(os.listdir(bspath + folder))

    class_id = {}

    if not os.path.exists(external_working_directory_path + 'user/datasets/' + dataset_name):
        os.mkdir(external_working_directory_path + 'user/datasets/' + dataset_name)

    with open(external_working_directory_path + 'user/datasets/' + dataset_name + '/labels.txt', 'a') as labels:
        writing_progress = bar.Bar("Writing image labels: ", max=n_images)
        for i, folder_name in enumerate(folders):
            for image in os.listdir(bspath+'/'+folder_name):
                labels.write(folder_name + '\n')
                writing_progress.next()
            class_id[i] = folder_name

    with open(external_working_directory_path + 'user/datasets/' + dataset_name + '/class_id.json', 'w') as txt:
        json.dump(class_id, txt, indent=3, sort_keys=True)


def write_paths(main_directory_path, dataset_name):
    bspath = external_working_directory_path+'user/'+main_directory_path+'/'

    folders = os.listdir(bspath)
    n_images = 0
    for folder in folders:
        n_images += len(os.listdir(bspath+folder))

    if not os.path.exists(external_working_directory_path + 'user/datasets/' + dataset_name):
        os.mkdir(external_working_directory_path + 'user/datasets/' + dataset_name)

    with open(external_working_directory_path + 'user/datasets/' + dataset_name + '/paths.txt', 'a') as paths:
        writing_progress = bar.Bar("Writing image paths: ", max=n_images)
        for n, folder in enumerate(folders):
            images = os.listdir(bspath+folder)
            for tp in images:
                paths.write(bspath+folder+'/'+tp+'\n')
                writing_progress.next()
    print("")
