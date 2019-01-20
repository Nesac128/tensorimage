import os
from progress import bar

from tensorimage.config.info import workspace_dir
from tensorimage.util.system.mkdir import mkdir


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
        for folder in folders:
            images = os.listdir(dataset_path+'/'+folder)
            for tp in images:
                paths.write(dataset_path+'/'+folder+'/'+tp+'\n')
                writing_progress.next()

