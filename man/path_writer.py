import os

from nnir.meta.config import external_working_directory_path

bspath = external_working_directory_path+'/testing_images/TwinFaces/'

folders = os.listdir(bspath)

if not os.path.exists(external_working_directory_path+'/datasets/TwinFaces'):
    os.mkdir(external_working_directory_path+'/datasets/TwinFaces')

with open(external_working_directory_path+'/datasets/TwinFaces/paths2.txt', 'a') as paths:
    for folder in folders:
        for image in os.listdir(bspath+folder):
            paths.write(bspath+folder+'/'+image+'\n')
