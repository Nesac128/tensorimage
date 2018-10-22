import os
import json
from progress import bar

from src.config import workspace_dir
from src.man.mkdir import mkdir


def write_labels(dataset_name):
    bspath = workspace_dir + 'user/training_images/' + dataset_name + '/'
    dataset_dir = workspace_dir + 'user/datasets/' + dataset_name
    mkdir(dataset_dir)

    folders = os.listdir(bspath)
    n_images = 0
    for folder in folders:
        n_images += len(os.listdir(bspath + folder))

    class_id = {}

    with open(dataset_dir + '/labels.txt', 'a') as labels:
        writing_progress = bar.Bar("Writing image labels: ", max=n_images)
        for i, folder_name in enumerate(folders):
            for image in os.listdir(bspath+'/'+folder_name):
                labels.write(folder_name + '\n')
                writing_progress.next()
            class_id[i] = folder_name

    with open(workspace_dir + 'user/datasets/' + dataset_name + '/class_id.json', 'w') as txt:
        json.dump(class_id, txt, indent=3, sort_keys=True)


def write_training_dataset_paths(dataset_name):
    bspath = workspace_dir+'user/training_images/'+dataset_name+'/'
    dataset_dir = workspace_dir + 'user/datasets/' + dataset_name
    mkdir(dataset_dir)

    folders = os.listdir(bspath)
    n_images = 0
    for folder in folders:
        n_images += len(os.listdir(bspath+folder))

    with open(dataset_dir + '/paths.txt', 'a') as paths:
        writing_progress = bar.Bar("Writing image paths: ", max=n_images)
        for n, folder in enumerate(folders):
            images = os.listdir(bspath+folder)
            for tp in images:
                paths.write(bspath+folder+'/'+tp+'\n')
                writing_progress.next()


def write_unclassified_dataset_paths(dataset_name):
    bspath = workspace_dir+'user/unclassified_images/'+dataset_name+'/'
    dataset_dir = workspace_dir+'user/datasets/'+dataset_name
    mkdir(dataset_dir)

    images = os.listdir(bspath)
    n_images = len(images)

    writing_progress = bar.Bar("Writing image paths: ", max=n_images)
    with open(dataset_dir+'/paths.txt', 'a') as paths:
        for image in images:
            paths.write(bspath+image)
            writing_progress.next()


