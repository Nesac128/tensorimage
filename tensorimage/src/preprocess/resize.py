import os
import cv2
from progress.bar import Bar

from src.config import workspace_dir


def resize_training_dataset(dataset_name,
                            size: tuple):
    folders = os.listdir(dataset_name)
    resize_progress = Bar('Resizing images: ', max=len(folders))
    for folder in folders:
        images = os.listdir(workspace_dir+'users/training_images/'+dataset_name+'/'+folder)
        for image in images:
            image_ = cv2.imread(workspace_dir+'user/training_images/'+dataset_name+'/'+folder+'/'+image)
            resized_image = cv2.resize(image_, size)
            cv2.imwrite(workspace_dir+'user/training_images/'+dataset_name+'/'+folder+'/'+image, resized_image)
        resize_progress.next()


def resize_unclassified_dataset(dataset_name,
                                size: tuple):
    images = os.listdir(dataset_name)
    resize_progress = Bar('Resizing images: ', max=len(images))
    for image in images:
        image_ = cv2.imread(workspace_dir+'user/unclassified_images/'+dataset_name+'/'+image)
        resized_image = cv2.resize(image_, size)
        cv2.imwrite(workspace_dir+'user/unclassified_images/'+dataset_name+'/'+image, resized_image)
        resize_progress.next()
