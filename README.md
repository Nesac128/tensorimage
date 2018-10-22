# TensorImage
TensorImage is a machine learning tool which provides an user friendly way to train image recognition models, by implementing Convolutional Neural Networks. Moreover, it offers the option to use these models to make predictions for thousands of unclassified images quickly and easily.

## Getting started
These are the steps you need to follow to get TensorImage working on your computer.
### Download TensorImage
You can now [download](https://github.com/TensorImage/TensorImage/releases) the latest TensorImage version.
### Install dependencies
You have to install the dependencies that are required by TensorImage. From the terminal:
```shell
# Access repository directory
$ cd TensorImage/

# Run install script
$ python3 setup.py
```
#### Open-source libraries used by TensorImage:
- [TensorFlow](https://github.com/tensorflow/tensorflow)

- [TensorBoard](https://github.com/tensorflow/tensorboard)

- [Click](https://github.com/pallets/click)

- [OpenCV Python](https://github.com/skvark/opencv-python)

- [Numpy](https://github.com/numpy/numpy)

- [Pandas](https://github.com/pandas-dev/pandas)

- [Pillow](https://github.com/python-pillow/Pillow)

- [Sci-kit learn](https://github.com/scikit-learn/scikit-learn)

- [Progress](https://github.com/Xfennec/progress)
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
#### Adding an unclassified image dataset
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
For ```model_folder_name``` enter the folder name where the trained model will be stored, and for ```model_filename``` pass the filename that will be used for the model files.
For ```learning_rate```, ```n_epochs``` and ```l2_regularization_beta```, pass the values that will be used as learning rate, the number of epochs and value used for L2 regularization (a technique used to prevent overfitting).

The rest of the parameters are options, and are not required, as they have default values. For ```optimizer``` the default value is ```GradientDescent```, and if you want to use the ```Adam``` optimizer, pass ```--optimizer Adam```. The default value for ```train_test_split``` is ```0.2```, and if you want to change the value, enter ```--train_test_split value```. The default value for ```batch_size``` is ```32```, and if you want to use a different value, pass ```--batch_size batch_size_value```. Finally, the ```augment_data``` option is a boolean which specifies if you want to augment the input training data automatically, or not. Its default value is ```False```, and if you want to use data augmentation, enter ```--augment_data True``` .

### Visualizing training progress with TensorBoard
From the terminal:
```shell
# Access TensorBoard log directory inside your workspace
$ cd workspace_dir/user/logs

$ tensorboard --logdir id_name
```
For ```id_name``` pass the ```id_name``` that you used for training, as it will be the name of the directory where the training progress information is stored. After that a link should appear in the terminal. Open it, and you should now have TensorBoard open.
### Classifying 
If you have already extracted the image data for an unclassified dataset, from the terminal:
```shell
$ cd tensorimage/tensorimage
$ python3 set.py classify
$ python3 main.py id_name* model_folder_name* model_name* training_dataset_name* --show_images [True/False]
```
__* required__
For ```id_name``` pass the ```id_name``` that you passed for extracting the image data. It will be used by TensorImage to identify and use the extracted image data for classification. For ```model_folder_name``` and ```model_name``` enter the ```model_folder_name``` and ```model_name``` you entered during the training process. It is the model that will be used for classification, where what TensorImage has learned about the training dataset is stored in. For ```training_dataset_name``` pass the training dataset name from which image data was extracted, and was used for the training process. The default value for ```show_images``` is ```False```. To set it to ```True```, just enter ```--show_images True```. If set to ```True```, the classifier will output the images individually, on a window, with the title being its predicted class (if classifying on many images, it is recommended to leave ```show_images``` as ```False```).

## License
TensorImage is licensed under the [GPL-2.0 license](https://github.com/TensorImage/TensorImage/blob/master/LICENSE.md).
