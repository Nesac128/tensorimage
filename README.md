# nnir
__nnir__ is a Python tool which uses machine-learning, and the [Tensorflow library](https://github.com/tensorflow/tensorflow) to provide an easy way to train a model from data, whether image data or raw data. Models, once trained, can later be used within the package to predict on unclassified data of any type. Works only for Linux and MacOS.

# Necessary preparations before using the tool
1. Extract the package after downloading, if necessary
2. Open terminal and access the main directory (nnir/), run __python3 setup.py__ from the main repository folder, __nnir/__ __Do not close the terminal__
3. Access config.py and modify the external_working_directory_path to a path where your workspace will be located. The workspace folder will later be created automatically
4. Run __python3 \__init__.py
 
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
![training_example_image](https://nesac128.github.io/nnir_readme_images/training_ex.jpg)
![cost_example_image](https://nesac128.github.io/nnir_readme_images/cost_ex.jpg)
![accuracy_example_image](https://nesac128.github.io/nnir_readme_images/accuracy_ex.jpg)

### 1. Writing paths and labels for images
#### From the main repository folder __nnir__:
- __python3 src/user/set.py write_paths__
- __python3 src/user/main.py training_images/your_dataset_name your_dataset_name__
- __python3 src/user/set.py write_labels__
- __python3 src/user/main.py training_images/your_dataset_name your_dataset_name__ again

Once you have executed all of the above, 3 files should have been created: 
- __your_workspace_folder_path/datasets/your_dataset_name/paths.txt__ containing the paths for all the images
- __your_workspace_folder_path/datasets/your_dataset_name/labels.txt__ containing the labels for all the images
- __your_workspace_folder_path/datasets/your_dataset_name/obj_labels.json__ containing the correspondence of program-generated labels and the names of each class. (Alphabetically ordered program-generated labels are used for the training process, instead of the class name, because it facilitates __nnir__ the process of matching the final numerical prediction to a label)

