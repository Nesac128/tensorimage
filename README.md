# TensorImage
TensorImage is a machine learning tool to train image recognition models, by implementing Convolutional Neural Networks. Moreover, it offers the possibility to use these models to make predictions for thousands of unclassified images quickly and easily.

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
workspace_dir = '/path/to/workspace/'
tensorimage_path = '/path/to/repository/'
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
For being able to add a training dataset to TensorImage, it must have the following structure:
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
$ python3 main.py data_name /path/to/training/dataset
```
##### Args:
- ```data_name```: a unique name assigned to a dataset used by TensorImage to identify image data
- ```/path/to/training/dataset```: path to the training dataset to add to TensorImage

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
$ python3 main.py data_name /path/to/unclassified/dataset
```
##### Args:
- ```data_name```: a unique name assigned to a dataset used by TensorImage to identify image data
- ```/path/to/unclassified/dataset```: path to the unclassified dataset to add to TensorImage

### Training
Assuming you have already extracted the image data for your training dataset, from the terminal:
```shell
$ cd tensorimage/tensorimage
$ python3 set.py train
$ python3 main.py data_name* training_name* learning_rate* n_epochs* l2_regularization_beta* --train_test_split train_test_split --batch_size batch_size --augment_data augment_data --cnn_architecture cnn_architecture
```
__* required__

##### Args:
- ```data_name```: data name that was assigned to training dataset that TensorImage will use for training
- ```training_name```: unique name assigned to a training operation. TensorImage will use it to identify the model files for classification
- ```learning_rate```: learning rate used for training
- ```n_epochs```: number of epochs
- ```l2_regularization_beta```: beta value used for L2 regularization to reduce overfitting


- ```train_test_split```: proportion of input data which TensorImage will use as testing set
- ```batch_size```: batch size
- ```augment_data```: ```True``` or ```False```, augment the input data or not
- ```cnn_architecture```: Convolutional Neural Network architecture that will be used. Available architectures:
    - AlexNet: to use the AlexNet architecture pass ```alexnet```. Input shape [227x227x3]. If image dimensions are not 227x227, they will be automatically resized. 
    - CNN model1: to use the CNN model1 architecture pass ```cnn_model1```. It accepts any image size (however all image sizes must be the same), original dimensions will be kept. It has been created by TensorImage for testing purposes and has achieved up to around 94% testing accuracy and around 99% training accuracy on a reduced version of the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database).

### Visualizing training progress with TensorBoard
From the terminal:
```shell
# Access TensorBoard log directory inside your workspace
$ cd workspace_dir/user/logs

$ tensorboard --logdir training_name
```
Args:
- ```training_name```: training name that was used for the training operation that you want to visualize
### Classifying 
If you have already extracted the image data for an unclassified dataset, from the terminal:
```shell
$ cd tensorimage/tensorimage
$ python3 set.py classify
$ python3 main.py data_name* training_name* classification_name* --show_images show_images
```
__* required__

##### Args:
- ```data_name```: data name that was assigned to unclassified dataset that TensorImage will use for classification
- ```training_name```: training name assigned to training operation, from where TensorImage will use the model to classify
- ```classification_name```: unique name assigned to the image classification operation
- ```show_images```: ```True``` or ```False```, display images with predicted labels after classification or not


## License
TensorImage is licensed under the [GPL-2.0 license](https://github.com/TensorImage/TensorImage/blob/master/LICENSE.md).
