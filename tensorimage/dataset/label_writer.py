import json
import os
from progress import bar

from tensorimage.config.info import workspace_dir
from tensorimage.util.system.mkdir import mkdir


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
