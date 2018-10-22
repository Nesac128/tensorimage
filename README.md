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
Modify ```workspace_dir``` to the workspace folder that you will be using for TensorImage. It is not necessary for you to create the folder, as it will be created automatically in another step. Modify ```tensorimage_path``` to the path where TensorImage has been saved. Now save and exit the configuration file.

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
+-- your_dataset  (directory)
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
Assuming you have already structured your dataset, from the terminal:
```shell
$ cd tensorimage/tensorimage/
$ python3 set.py add_training_dataset
$ python3 main.py path/to/your/training/dataset
```
#### Bulk resizing images in training dataset
From the terminal:
```shell
$ cd tensorimage/tensorimage/

# Set option for training dataset resizing
$ python3 set.py resize_training_dataset

# Show help (optional)
$ python3 main.py --help
# Will output:
# Usage: main.py [OPTIONS] DATASET_NAME WIDTH HEIGHT

$ python3 main.py dataset_name output_image_width output_image_height
```
#### Structuring an unclassified image dataset
Your unclassified image dataset has to have the following structure:
```
+-- your_dataset  (directory)
   |   image1.jpg  (image)
   |   image2.jpg  (image)
   |   image3.jpg  (image)
   |   image4.jpg  (image)
   |   image5.jpg  (image)
   |   image6.jpg  (image)
   |   ...         (rest of images)
```
#### Adding and unclassified image dataset
Again, assuming you have correctly structured your unclassified image dataset, from the terminal:
```shell
$ cd tensorimage/tensorimage
$ python3 set.py add_unclassified_dataset
$ python3 main.py path/to/your/unclassified/dataset
```
#### Bulk resizing images in an unclassified image dataset
From the terminal:
```shell
$ cd tensorimage/tensorimage/

# Set option for training dataset resizing
$ python3 set.py resize_unclassified_dataset

# Show help (optional)
$ python3 main.py --help
# Will output:
# Usage: main.py [OPTIONS] DATASET_NAME WIDTH HEIGHT

$ python3 main.py dataset_name output_image_width output_image_height
```
### Extracting image data from training dataset
From the terminal:
```shell
$ cd tensorimage/tensorimage
$ python3 set.py write_training_dataset_data
$ python3 main.py dataset_name output_filename.csv id_name
```
You will probably have understood what ```dataset_name``` and ```output_filename``` mean, but not perhaps for ```id_name```. ```id_name``` is just a __unique__ name that you will give to the extracted data, which will be used to later specify what image data to use for training.
### Extracting image data from unclassified dataset
From the terminal:
```shell
$ cd tensorimage/tensorimage
$ python3 set.py write_unclassified_dataset_data
$ python3 main.py dataset_name output_filename.csv id_name
```
If cannot understand what ```id_name``` is used for, and what you have to pass as a parameter, read the last part of the section above.
### Training
Assuming you have already extracted the image data for your training dataset, from the terminal:
```shell
$ cd tensorimage/tensorimage
$ python3 set.py train
$ python3 main.py id_name* model_folder_name* model_filename* learning_rate* n_epochs* l2_regularization_beta* --optimizer [Adam/GradientDescent] --train_test_split train_test_split --batch_size batch_size --augment_data [True/False]
```
__* required__

For ```id_name```, pass the ```id_name``` you have entered in one of the previous steps, when extracting the image data. This is used by TensorImage to identify exactly what data is going to be used for training.
For ```model_folder_name``` enter the folder name where the trained model will be stored, and for ```model_filename``` pass the filename that will be used for the actual model files.

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
