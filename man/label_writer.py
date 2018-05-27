import os
import string
from itertools import product
import json

from meta.config import external_working_directory_path

bspath = external_working_directory_path+'/training_images/TwinFaces/'

folders = os.listdir(bspath)

keys = []

for k in product(string.ascii_uppercase, repeat=2):
    keys.append(k[0]+k[1])

ukeys = []

for kn in range(3):
    ukeys.append(keys[kn])


obj_to_lb = {}

if not os.path.exists(external_working_directory_path+'/datasets/TwinFaces'):
    os.mkdir(external_working_directory_path+'/datasets/TwinFaces')

with open(external_working_directory_path+'/datasets/TwinFaces/labels.txt', 'a') as labels:
    i = 0
    for folder in folders:
        for image in os.listdir(bspath+folder):
            labels.write(ukeys[i]+'\n')
        obj_to_lb[folder.split('-')[0]] = ukeys[i]
        i += 1

with open(external_working_directory_path+'/datasets/TwinFaces/obj_labels.json', 'w') as txt:
    json.dump(obj_to_lb, txt, indent=3, sort_keys=True)
