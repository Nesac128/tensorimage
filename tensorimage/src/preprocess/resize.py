import os
import cv2


def resize(image_containing_folder_path,
           new_folder_path,
           size: tuple):
    images = os.listdir(image_containing_folder_path)
    for img in images:
        image = cv2.imread(image_containing_folder_path+'/'+img)
        resized_image = cv2.resize(image, size)
        cv2.imwrite(new_folder_path+'/'+img, resized_image)
