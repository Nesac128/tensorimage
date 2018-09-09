# nnir
__nnir__ is a Python tool which uses machine-learning, and the [Tensorflow library](https://github.com/tensorflow/tensorflow) to provide an easy way to train a model from data, whether image data or raw data. Models, once trained, can later be used within the package to predict on unclassified data of any type. Works only for Linux and MacOS.

# Necessary preparations before using the tool
1. Extract the package after downloading, if necessary
2. Open terminal and access the main repository folder (__nnir/__), and run __python3 setup.py__ from the main repository folder, __nnir/__ . __Do not close the terminal__
3. Access __nnir/src/config.py__ and modify the external_working_directory_path to a path where your workspace will be located. The workspace folder will later be created automatically
4. Run __python3 \__init\__.py__
 
 # Documentation
 ## 1. Preparing a dataset
 In order to be able to train a machine-learning model you must have an image dataset within a folder in your workspace folder, inside the training_images/ folder. The image dataset for training __must__ have the following structure (the displayed structure is based on your workspace folder as root):
 
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
You can either use your own image dataset or download one from the internet, from websites such as:
- [https://www.kaggle.com](https://www.kaggle.com)
- [UCI machine-learning repository](https://archive.ics.uci.edu/ml/index.php)
- [http://deeplearning.net/datasets/](http://deeplearning.net/datasets/)
#### Note!
- You may need to adapt the structure of any dataset, whether yours or downloaded from the internet
- The current nnir version only accepts images with __.jpg__ format
 
## 2. Training
### 1. Writing paths and labels for images
#### From the main repository folder __nnir/__:
- __python3 src/set.py write_paths__
- __python3 src/main.py training_images/your_dataset_name your_dataset_name__
- __python3 src/set.py write_labels__
- __python3 src/main.py training_images/your_dataset_name your_dataset_name__ again

Once you have executed all of the above, 3 files should have been created: 
- __your_workspace_folder_path/datasets/your_dataset_name/paths.txt__ containing the paths for all the images
- __your_workspace_folder_path/datasets/your_dataset_name/labels.txt__ containing the labels for all the images
- __your_workspace_folder_path/datasets/your_dataset_name/obj_labels.json__ containing the correspondence of program-generated labels and the names of each class. (Alphabetically ordered program-generated labels are used for the training process, instead of the class name, because it facilitates __nnir__ the process of matching the final numerical prediction to a label)

### 2. Extracting data from images for training
#### From the main repository folder __nnir/__:
- __python3 nnir/set.py im_man1__ (im_man1 is the option to extract data from images for training, together with labels)
- __python3 nnir/main.py your_dataset_name output_file_name.csv__ (__only__ put a file name, __not__ a path, as the output file will automatically be stored in __your_workspace_folder/data/training/your_dataset_name/output_file_name.csv__)
A new folder, whose name should be a number, should have been created in __nnir/src/meta/sess/__, which contains __impaths.csv__ (contains a copy of the image dataset paths) and __meta.txt__ with information necessary for the training process.

### 3. Training the data with a Convolutional Neural Network
#### From the main repository folder __nnir/__:
- __python3 nnir/set.py train__ (set option to train)
- __python3 __nnir/main.py id output_model_name output_model_filename__ :  __id__ is a number containing all the information for the data you want to train, it starts from __0__. To see the number of __id__ s you have, you can access __nnir/src/meta/sess__ . __output_model_name__ is the folder name for the output trained model. __output_model_filename__ is a name (do not enter a format, just a name) for the many files contained in the model. All of these are required parameters. Optional parameters are:
- __--display_frequency display_frequency_integer__ (display_frequency_integer shows the progress in accuracy and loss. Default: __50__)
- __--n_epochs number_of_epochs__ (__number_of_epochs__ is the number of training epochs. Default: __600__)
- __--learning_rate float_value__ (__float_value__ is a float that defines the learning rate. Default: __0.00000008__)
- __--train_test_split train_test_split_float__ (__train_test_split_float__ is a float which defines the proportion of data used as training set and testing set, e.g: __0.4__ means that 40% of the data will be used as testing set. Default: __0.1__)
- __--l2_regularization_beta l2_reg_beta_float__ (__l2_reg_beta_float__ is a float that defines the beta for __L2 regularization__ . Default: __0.01__)

![training_example_image](https://nesac128.github.io/nnir_readme_images/training_ex.jpg)
![cost_example_image](https://nesac128.github.io/nnir_readme_images/cost_ex.jpg)
![accuracy_example_image](https://nesac128.github.io/nnir_readme_images/accuracy_ex.jpg)
