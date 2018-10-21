# TensorImage
TensorImage is a machine learning tool which provides an user friendly way to train image recognition models, by implementing Convolutional Neural Networks. Moreover, it offers the option to use these models to make predictions for thousands of unclassified images quickly and easily.

## Getting started
These are the steps you need to follow to get TensorImage working on your computer.
### Download TensorImage
From the terminal:
```shell
$ git clone https://github.com/TensorImage/TensorImage.git
```
### Install dependencies
You have to install the dependencies that are required by TensorImage. From the terminal:
```shell
# Access repository directory
$ cd TensorImage/
# Run install script
$ python3 setup.py
```
### Configure TensorImage
In order to get TensorImage working, you must adjust the configuration to your computer. From the terminal:
```shell
# Access repository directory
$ cd TensorImage/

# Open configuration file for editing
$ nano config.py
```
You should now have a terminal similar to the following:
```python
# User configurations
workspace_dir = 'path/to/workspace/'
tensorimage_path = 'path/to/repository'
...
```
Modify ```workspace_dir``` to the workspace folder that you will be using for TensorImage. It is not necessary for you to create the folder, as it will be created automatically in another step. Modify ```tensorimage_path``` to the path where TensorImage has been saved. Now save and exit the configuration file. Press:
```shell
Ctrl+X
y # (y means yes, to save the file)
Enter
```
To finish up with setting up TensorImage, from the terminal:
```shell
# Access repository directory
$ cd TensorImage/

# Run __init__.py
$ python3 __init__.py
```
You are now ready to begin using TensorImage!

 ## Usage
 ### Preprocesing a dataset
 #### Structuring a training dataset
 In order to be able to train an image classification model, you must have the image dataset inside ```workspace_dir/training_images/```. The training images dataset must have the following structure:
```
 +-- training_images  (directory)
 |   +-- your_dataset  (directory)
     |   +-- class1  (directory)
         |   image1.jpg  (image)
         |   image2.jpg  (image)
         |   image3.jpg  (image)
         |   ...         (rest of images)
         
     |   +-- class2  (directory)
         |   image1.jpg  (image)
         |   image2.jpg  (image)
         |   image3.jpg  (image)
         |   ...         (rest of images)
         
     |   +-- ...  (rest of directories)
         |   image1.jpg  (image)
         |   image2.jpg  (image)
         |   image3.jpg  (image)
         |   ...         (rest of images)
```
#### Adding a training dataset to TensorImage
Assuming you have already downloaded and structured your training images dataset, to add a training dataset to TensorImage, you just need to move (or copy) it to ```workspace_dir/user/training_images/```.
#### Bulk resizing images in training dataset
From the terminal:
```shell
$ cd TensorImage/TensorImage/

# Set option for training dataset resizing
$ python3 set.py resize

# Show help (optional)
$ python3 main.py --help
# Will output:
# Usage: main.py [OPTIONS] DATASET_NAME WIDTH HEIGHT

$ python3 main.py your_training_dataset_name output_image_width output_image_height
```
 
## 4.2 Training
### 4.2.1 Automatically writing image paths and labels
#### From the nnir/ directory, run:
```
python3 set.py write_paths
python3 main.py training_images/your_dataset_name your_dataset_name
python3 set.py write_labels
python3 main.py training_images/your_dataset_name your_dataset_name
```
Once you have run all of the above, 3 files should have been created inside ```your_workspace_folder_path/datasets/your_dataset_name/```: 
```
paths.txt (contains the paths for every image in the dataset)
labels.txt (contains the labels for every image in the dataset)
obj_labels.json (contains program-generated labels matched to image labels taken from directory name for each image class)
```

### 4.2.2  Extracting image data for training
#### From the nnir/ directory, run:
```
python3 set.py im_man1
python3 main.py your_dataset_name output_filename.csv
```
The output CSV file containing all of the image data together with the labels will be stored automatically in ```your_workspace_folder_path/data/training/your_dataset_name/output_filename.csv```

### 4.2.3 Training
#### From the nnir/ directory, run:
 ```
 python3 main.py data_id* model_directory_name* model_filename* --learning_rate learning_rate --n_epochs n_epochs --display_frequency display_frequency --train_test_split train_test_split --l2_regularization_beta l2_regularization_beta
 ```
 #### Parameter information:
```
data_id::  type: number    required: yes    info: a number which NNIR uses to find internally generated information about data that has been extracted for training

model_directory_name::  type: string    required: yes    info: folder name where trained model will be stored

model_filename::  type: string    info: filename for model which will be stored in model_directory_name

learning_rate::  type: float    required: no    default: 0.0000000008    info: learning rate for training process

n_epochs::  type:  integer    required: no    default: 2500    info: number of epochs for training process

display_frequency:  type: integer    required: no    default: 50    info: every how many epochs information regarding training/testing accuracy and training/testing cost will be displayed in graph

train_test_split:  type: float    required: no    default: 0.1    info: proportion of input data that will be used as testing set
 
l2_regularization_beta:  type: float    required: no    default: 0.01    info: float which will be used for L2 regularization to reduce model overfitting
```

![training_example_image](https://nesac128.github.io/nnir_readme_images/training_ex.jpg)
![cost_example_image](https://nesac128.github.io/nnir_readme_images/cost_ex.jpg)
![accuracy_example_image](https://nesac128.github.io/nnir_readme_images/accuracy_ex.jpg)
