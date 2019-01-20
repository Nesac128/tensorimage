import tensorflow as tf
import numpy as np
import csv
import ast

from tensorimage.config.info import *
from tensorimage.image.display import display_image
from tensorimage.file.reader import *
from tensorimage.util.system.mkdir import mkdir
from tensorimage.classify.restore_model import ModelRestorer
from tensorimage.base.models.map.model import model_map


class Classifier:
    def __init__(self,
                 data_name,
                 training_name,
                 classification_name,
                 show_images: tuple = (False, 10)):
        """
        :param data_name: name which was used as data_name when extracting the unclassified data from image dataset;
        used to identify the data to classify
        :param training_name: name which was used as training_name when training an image classification model; used to
        identify the model for making predictions
        :param classification_name: unique name assigned to a specific classification operation; used as directory name
        in workspace where predictions will be stored
        :param show_images: tuple containing a boolean (display images with predicted classes or not) and the maximum
        number of image to display
        """
        self.data_name = data_name
        self.training_name = training_name
        self.classification_name = classification_name
        try:
            self.show_images = ast.literal_eval(str(show_images[0]))
            self.max_images = show_images[1]
        except NameError:
            self.show_images = False
            self.max_images = show_images[1]

        self.raw_predictions = []
        self.predictions = {}

        # Read unclassified data metadata
        self.metadata_reader = JSONReader(self.data_name, dataset_metafile_path)
        self.metadata_reader.bulk_read()
        self.metadata_reader.select()
        image_metadata = self.metadata_reader.selected_data
        self.data_path = image_metadata["data_path"]
        self.width = image_metadata["width"]
        self.height = image_metadata["height"]
        self.path_file = image_metadata["path_file"]
        self.trainable = image_metadata["trainable"]
        if ast.literal_eval(self.trainable):
            raise AssertionError("Data is trainable")

        # Read trained model metadata
        self.training_metadata_reader = JSONReader(self.training_name, training_metafile_path)
        self.training_metadata_reader.bulk_read()
        self.training_metadata_reader.select()
        training_metadata = self.training_metadata_reader.selected_data
        self.model_path = training_metadata["model_folder_name"]
        self.model_name = training_metadata["model_name"]
        self.training_dataset_name = training_metadata["dataset_name"]
        self.n_classes = training_metadata["n_classes"]
        self.model_folder_name = self.model_path.split('/')[-1]
        self.architecture = training_metadata["architecture"]

        # Define prediction storage paths
        self.raw_predictions_path = workspace_dir+'user/predictions/'+self.model_folder_name+'/'+self.classification_name + \
            '/raw_predictions_.csv'
        self.predictions_paths_path = workspace_dir+'user/predictions/'+self.model_folder_name+'/'+self.classification_name + \
            '/predictions_to_paths.csv'

        self.csv_dataset_reader = CSVReader(self.data_path)
        self.class_id_reader = JSONReader(None, workspace_dir+'user/training_datasets/' +
                                          self.training_dataset_name + '/class_id.json')

        self.path_reader = TXTReader(self.path_file)
        self.path_reader.read_raw()
        self.path_reader.parse()
        self.image_paths = self.path_reader.parsed_data

        self.class_id_reader.bulk_read()
        self.class_id = self.class_id_reader.bulk_data

        self.X = None
        self.X_ = None

        self.sess = tf.Session()
        self.model_restorer = ModelRestorer(self.model_path, self.model_name, self.architecture, self.sess)

        self.n_images = 0

    def build_dataset(self):
        self.csv_dataset_reader.read_file()
        self.X = self.csv_dataset_reader.X
        self.X_ = np.ndarray.tolist(self.X)
        self.n_images = len(self.X)

    def predict(self):
        self.model_restorer.start()

        x = tf.placeholder(tf.float32, [None, self.height, self.width, 3])

        convnet = model_map[self.architecture](x, self.n_classes)
        model = convnet.convnet()
        self.sess.run(tf.global_variables_initializer())

        self.X = self.sess.run(tf.reshape(self.X, [self.X.shape[0], self.height, self.width, 3]))
        predictions = self.sess.run(model, feed_dict={x: self.X})
        self.raw_predictions = np.ndarray.tolist(self.sess.run(tf.argmax(predictions, 1)))
        self._match_class_id()
    
    def write_predictions(self):
        mkdir(base_predictions_store_path)
        mkdir(base_predictions_store_path + '/' + self.model_folder_name)
        mkdir(base_predictions_store_path + '/' + self.model_folder_name + '/' + self.classification_name)

        for image_n in range(self.n_images):
            self.X_[image_n] = np.append(self.X_[image_n], self.predictions[self.image_paths[image_n]])

        with open(self.raw_predictions_path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for image_prediction in self.X_:
                writer.writerow(image_prediction)

        with open(self.predictions_paths_path, 'w') as pathfile:
            writer = csv.writer(pathfile, delimiter=',')
            for n in range(self.n_images):
                writer.writerow([self.image_paths[n], self.predictions[self.image_paths[n]]])
        self.image_paths = np.asarray(self.image_paths)
        np.random.shuffle(self.image_paths)
        if self.show_images:
            for n in range(self.max_images):
                display_image(self.predictions[self.image_paths[n]], self.image_paths[n])

    def _match_class_id(self):
        for n, fp in enumerate(self.raw_predictions):
            self.predictions[self.image_paths[n]] = self.class_id[str(fp)]
