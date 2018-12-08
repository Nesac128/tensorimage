import os
import json
from progress import bar

from tensorimage.src.config import workspace_dir
from tensorimage.src.os.mkdir import mkdir


def write_unclassified_dataset_paths(dataset_path, dataset_name):
    dataset_dir = workspace_dir+'user/unclassified_datasets/'+dataset_name
    mkdir(dataset_dir)

    images = os.listdir(dataset_path)
    n_images = len(images)

    writing_progress = bar.Bar("Writing image paths: ", max=n_images)
    with open(dataset_dir+'/paths.txt', 'a') as paths:
        for image in images:
            paths.write(dataset_path+'/'+image+'\n')
            writing_progress.next()


def write_training_dataset_paths(dataset_path, dataset_name):
    dataset_dir = workspace_dir + 'user/training_datasets/' + dataset_name
    mkdir(dataset_dir)
    folders = os.listdir(dataset_path)
    n_images = 0
    for folder in folders:
        n_images += len(os.listdir(dataset_path+'/'+folder))

    with open(dataset_dir + '/paths.txt', 'a') as paths:
        writing_progress = bar.Bar("Writing image paths: ", max=n_images)
        for n, folder in enumerate(folders):
            images = os.listdir(dataset_path+'/'+folder)
            for tp in images:
                paths.write(dataset_path+'/'+folder+'/'+tp+'\n')
                writing_progress.next()


def write_labels(dataset_path, dataset_name):
    dataset_dir = workspace_dir + 'user/training_datasets/' + dataset_name
    mkdir(dataset_dir)

    folders = os.listdir(dataset_path)
    n_images = 0
    for folder in folders:
        n_images += len(os.listdir(dataset_path+'/'+folder))

    class_id = {}

    with open(dataset_dir + '/labels.txt', 'a') as labels:
        writing_progress = bar.Bar("Writing image labels: ", max=n_images)
        for i, folder_name in enumerate(folders):
            for image in os.listdir(dataset_path+'/'+folder_name):
                labels.write(folder_name + '\n')
                writing_progress.next()
            class_id[i] = folder_name

    with open(dataset_dir + '/class_id.json', 'w') as txt:
        json.dump(class_id, txt, indent=3, sort_keys=True)
