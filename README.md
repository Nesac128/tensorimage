# 1 nnir
__nnir__ is a machine learning tool which uses Python, and [Tensorflow](https://github.com/tensorflow/tensorflow) to provide an user-friendly way to train models for image classification by implementing Convolutional Neural Networks. Moreover, it offers the possibility to use these models for image classification on ___any___ images in a fast and easy manner. 

# 2 Prerequisites
- Python 3.* for [Linux/UNIX](https://www.python.org/downloads/source/) or [Mac OS X](https://www.python.org/downloads/mac-osx/)
- Node.js for [Linux/UNIX](https://nodejs.org/en/download/package-manager/#debian-and-ubuntu-based-linux-distributions) or [MAC OS X](https://nodejs.org/en/download/package-manager/#macos)
- Install all of the dependencies by running __setup.py__
- Open __config.py__ and modify __external_working_directory_path__ to a folder path where your workspace will be located. Also modify __nnir_path__ to the path where you will save the downloaded repository
```
python3 __init__.py
```

# 3 Operations
There are several operations that can be carried out within __nnir__:
- Training OpCode: __train__

- Image classification OpCode: __classify__

- Extracting image data for training OpCode: __im_man1__

- Extracting image data for classifying OpCode: __im_man2__

- Writing image paths OpCode: __write_paths__

- Writing image labels OpCode: __write_labels__

- Bulk resizing images OpCode: __resize__

- Bulk changing format to __JPG__
 
 __OpCode = Operation Code__

 # 4 Documentation
 ## 4.1 Preprocesing a dataset
 In order to be able to train a model you must have the image dataset inside ```your_workspace_folder_path/training_images/```. The training image dataset __must__ have the following structure:
 
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
#### Notes
- The image names displayed inside each image class folder are just examples, the image names used by your dataset can be entirely different
- The current nnir version only accepts images with __JPG__ format


You can either use your own image dataset or download one from the internet, from websites such as:
- [https://www.kaggle.com](https://www.kaggle.com)
- [UCI machine-learning repository](https://archive.ics.uci.edu/ml/index.php)
- [http://deeplearning.net/datasets/](http://deeplearning.net/datasets/)

 
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
