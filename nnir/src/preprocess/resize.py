import os
import cv2


def resize(path, new_path, size: tuple):
    images = os.listdir(path)
    print(images)
    for img in images:
        print("lol")
        image = cv2.imread(path+'/'+img)
        print(image.shape)
        resized_image = cv2.resize(image, size)
        cv2.imwrite(new_path+'/'+img, resized_image)


# resize('/media/planetgazer8360/NASER/External Tensorflow workspace/training_images/CatsAndDogs/cats/')
