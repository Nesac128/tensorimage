__TensorImage__ is and open source package for image classification. There is a wide range of data augmentation operations that can be performed over training data to prevent overfitting and increase testing accuracy. It is easy to use and manage as all files, trained models and data are organized within a workspace directory, which you can change at any time in the configuration file, therefore being able have an indefinite amount of workspace directories for different purposes. Moreover, TensorImage can also be used to classify on thousands of images with trained image classification models. 

# Installation
## Download TensorImage
You can download the latest TensorImage version [here](https://github.com/TensorImage/TensorImage/releases).
## Installing dependencies
From the terminal:
```shell
# Access repository directory
$ cd TensorImage/
$ pip3 install -r requirements.txt
```
### Dependencies
- [TensorFlow](https://github.com/tensorflow/tensorflow)

- [TensorBoard](https://github.com/tensorflow/tensorboard)

- [OpenCV Python](https://github.com/skvark/opencv-python)

- [Numpy](https://github.com/numpy/numpy)

- [Pandas](https://github.com/pandas-dev/pandas)

- [Pillow](https://github.com/python-pillow/Pillow)

- [Sci-kit learn](https://github.com/scikit-learn/scikit-learn)

- [Progress](https://github.com/Xfennec/progress)

- [Scipy](https://github.com/scipy/scipy)

## Configure TensorImage
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
Modify ```workspace_dir``` to the workspace directory that you will be using for TensorImage. It is not necessary for you to create the folder, as it will be created automatically in another step. Modify ```tensorimage_path``` to the path where TensorImage has been saved. Now save and exit the configuration file.

# Dataset structures
## Training datasets
All training datasets must have the following structure:
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
## Unclassified datasets
All unclassified datasets must have the following structure:
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
# Examples
## Creating a workspace directory
Assuming you have the workspace directory set in ```config.py```:
```python
import tensorimage as ti

ti.src.make_workspace()
```
## Adding a training image dataset
```python
import tensorimage as ti

dataset_path = '/home/user/My training datasets/MNIST/' # Path to training dataset with images, should have structure as specified in the previous section
dataset_name = 'MNIST_training_images'
data_name = 'MNIST_training_data_1' # Unique name assigned to the specific set of data that will be created by running this code once. It will be used later to specify what data to use for training

ti.src.image.label_path_writer.write_training_dataset_paths(dataset_path, dataset_name)
ti.src.image.label_path_writer.write_labels(dataset_path, dataset_name)
image_loader = ti.src.image.loader.ImageLoader(data_name, dataset_name, 'training')
image_loader.extract_image_data()
image_writer = ti.src.image.writer.TrainingDataWriter(image_loader.image_data, data_name, dataset_name, image_loader.img_dims)
image_writer.write_image_data()
```

## Adding an unclassified image dataset
```python
import tensorimage as ti

dataset_path = '/home/user/My unclassified datasets/MNIST/'
dataset_name = 'MNIST_unclassified_images'
data_name = 'MNIST_unclassified_data_1'

ti.src.image.label_path_writer.write_unclassified_dataset_paths(dataset_path, dataset_name)
image_loader = ti.src.image.loader.ImageLoader(data_name, dataset_name, 'unclassified')
image_loader.extract_image_data()
image_writer = ti.src.image.writer.DataWriter(image_loader.image_data, data_name, dataset_name, image_loader.img_dims)
image_writer.write_image_data()
```

## Training
### Without data augmentation
```python
import tensorimage as ti

data_name = 'MNIST_training_data_1' # data_name assigned to extracted data previously
training_name = 'MNIST_train_op_1' # Unique name for 1 specific training operation that will be used to identify trained models and other information for classification
n_epochs = 600
learning_rate 0.08
l2_regularization_beta = 0.05 # Beta value for L2 Regularization (to prevent overfitting)
architecture = 'RosNet' # Other CNN architectures are also available
batch_size = 32
train_test_split = 0.2

trainer = ti.src.trainer.Train(data_name, training_name, n_epochs, learning_rate, l2_regularization_beta, architecture, data_augmentation_builder=(None, False), batch_size=batch_size, train_test_split=train_test_split)
trainer.build_dataset()
trainer.train()
trainer.store_model()
```

### With data augmentation
```python
import tensorimage as ti

data_name = 'MNIST_training_data_1' # data_name assigned to extracted data previously
training_name = 'MNIST_train_op_1' # Unique name for 1 specific training operation that will be used to identify trained models and other information for classification
n_epochs = 600
learning_rate 0.08
l2_regularization_beta = 0.05 # Beta value for L2 Regularization (to prevent overfitting)
architecture = 'RosNet' # Other CNN architectures are also available
batch_size = 32
train_test_split = 0.2
```
There are many data augmentation operations which you can perform on the training data. You can apply all of them to your training data, or just one, or none. You must pass the operation classes, with any required parameters, to the ```DataAugmentationBuilder()``` class, which will then be passed to the ```Train()``` class. The script continues below:
```python
# Image flipping
image_flipper_op = ti.src.data_augmentation.data_augmentation_ops.FlipImages()

# Salt-pepper noise
salt_vs_pepper = 0.1
amount = 0.0004
pepper_salt_noise_op = ti.src.data_augmentation.data_augmentation_ops.AddSaltPepperNoise(salt_vs_pepper=salt_vs_pepper, amount=amount)

# Lighting modification
max_delta = 0.8
lighting_modification_op = ti.src.data_augmentation.data_augmentation_ops.ModifyLighting(max_delta)

# Image rotation
image_rotation_op = ti.src.data_augmentation.data_augmentation_ops.RotateImages(10,20,30,40,50,60,70,80,90,100) # Parameters are *args specifying on which angles to rotate images

data_augmentation_builder = ti.src.data_augmentation.data_augmentation_builder.DataAugmentationBuilder(image_flipper_op, pepper_salt_noise_op, lighting_modification_op, image_rotation_op)

trainer = ti.src.trainer.Train(data_name, training_name, n_epochs, learning_rate, l2_regularization_beta, architecture, data_augmentation_builder=(data_augmentation_builder, True), batch_size=batch_size, train_test_split=train_test_split)
trainer.build_dataset()
trainer.train()
trainer.store_model()
```
The trained image classification model will be stored in the path:
```shell
workspace_dir/user/trained_models/training_name
```

### Available architectures
The available architectures that can be passed to the ```Train()``` class `architecture` parameter are:
- RosNet
- [AlexNet](https://en.wikipedia.org/wiki/AlexNet)

### Visualizing training progress with TensorBoard
```python
from tensorboard import default
from tensorboard import program
from tensorimage.config import workspace_dir

training_name = 'MNIST_train_op_1' # training_name that was used in training operation to visualize
log_directory = workspace_dir + 'user/logs/' + training_name 

tb = program.TensorBoard(default.PLUGIN_LOADERS, default.get_assets_zip_provider())
tb.configure(argv=['--logdir', log_directory])
tb.main()
```
## Training clusters
TensorImage can also be used to perform multiple training operations at once on different CPUs, only storing the models based on the final testing accuracy, which is helpful for feature engineering, as it will yield the top performers with the hyperparameters that were used.
```python
import tensorimage as ti
from tensorimage.trainer import *
from tensorimage.src.data_augmentation_ops import *
from tensorimage.src.data_augmentation_builder import DataAugmentationBuilder


data_name = 'MNIST_training_data_1' # data_name assigned to extracted data previously # data_name is the same for all training_names

# Training operation 1 (without augmentation)
training_name_1 = 'MNIST_train_op_1' # training_name assigned to this specific training operation
n_epochs_1 = 600
learning_rate_1 = 0.08
l2_regularization_beta_1 = 0.05 # Beta value for L2 Regularization (to prevent overfitting)
architecture_1 = 'RosNet' # Other CNN architectures are also available
batch_size_1 = 32
train_test_split_1 = 0.2

trainer1 = ti.src.trainer.Train(data_name, training_name_1, n_epochs_1, learning_rate_1, l2_regularization_beta_1, architecture_1, data_augmentation_builder=(None, False), batch_size=batch_size_1, train_test_split=train_test_split_1)

# Training operation 2 (with data augmentation)
training_name_2 = 'MNIST_train_op_2'
n_epochs_2 = 1200
learning_rate_2 = 0.09
l2_regularization_beta_2 = 0.08 # Beta value for L2 Regularization (to prevent overfitting)
architecture_2 = 'RosNet' # Other CNN architectures are also available
batch_size_2 = 32
train_test_split_2 = 0.3

# Building data augmentation operations
# Pepper-salt noise
salt_vs_pepper = 0.1
amount = 0.0004
pepper_salt_noise_op = AddPepperSaltNoise(salt_vs_pepper=salt_vs_pepper, amount=amount)

# Image rotation
image_rotation_op = RotateImages(5,10,15,20,25,30)

data_augmentation_builder = ti.src.data_augmentation.data_augmentation_builder.DataAugmentationBuilder(pepper_salt_noise_op, image_rotation_op)

trainer2 = ti.src.trainer.Train(data_name, training_name_2, n_epochs_2, learning_rate_2, l2_regularization_beta_2, architecture_2, data_augmentation_builder=(data_augmentation_builder, True), batch_size=batch_size_2, train_test_split=train_test_split_2)

cluster_trainer = ti.src.trainer.ClusterTrain(trainer1=trainer1, trainer2=trainer2)
cluster_trainer.train()
results = cluster_trainer.get_results(top_n=1)
print(results)
```
The ```top_n``` parameter means how many top performers ```cluster_trainer.get_results()``` will yield, which in this case is 1, as only 2 trainers with different hyperparameters (and data augmentation or not) are being compared. However, if you are comparing 10 different trainers, you may want to receive the top 3 performers based on testing accuracy. The output should be the following:
```python
{"trainer1": {
   "completed": True,
   "testing_accuracy": 0.97, # Final testing accuracy
   "testing_cost": 38, # Final testing cost
   "n_epochs": 1000, # Epochs used by this trainer
   "learning_rate": 0.09, # Learning rate used
   "l2_regularization_beta": 0.08 # L2 Regularization beta value used
   "train_test_split": 0.3, # Train-test split used
   "architecture": "RosNet" # ConvNet architecture used
   "batch_size": 32 # Batch size used
   }}
```
## Classification 
```python
import tensorimage as ti

data_name = 'MNIST_unclassified_data_1' # data_name assigned to extracted data from MNIST unclassified dataset
training_name = 'MNIST_train_op_1' # training_name assigned to training operation, will be used to identify the trained model
classification_name = 'MNIST_classify_op_1' # Unique name assigned to this specific classification operation
show_images = (True, 20) # Specifies if images with labels will be displayed, and the maximum amount of images to display

classifier = ti.src.classifier.Classifier(data_name, training_name, classification_name, show_images=show_images)
classifier.build_dataset()
classifier.predict()
classifier.write_predictions()
```
The final predictions for all of the unclassified images will be stored in the path:
```shell
workspace_dir/user/predictions/training_name/classification_name/
```
# License
TensorImage is licensed under the [GPL-3.0 license](https://github.com/TensorImage/TensorImage/blob/master/LICENSE.md).
