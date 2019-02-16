import tensorflow as tf
import numpy as np
import csv
import ast

from tensorimage.config.info import *
from tensorimage.image.display import display_image
from tensorimage.file.reader import *
from tensorimage.util.system.mkdir import mkdir
from tensorimage.base.models.map.model import model_map
from tensorimage.base.model import Model
from tensorimage.base.metadata_reader import TrainerMetadata, UnclassifiedDatasetMetadata


class Classifier(TrainerMetadata, UnclassifiedDatasetMetadata):
    def __init__(self,
                 data_name: str,
                 training_name: str,
                 classification_name: str,
                 show_images: tuple = (False, 10),
                 n_threads: int = 10):
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
        TrainerMetadata.__init__(self, training_name)
        UnclassifiedDatasetMetadata.__init__(self, data_name)

        self.data_name = data_name
        self.training_name = training_name
        self.classification_name = classification_name
        try:
            self.show_images = ast.literal_eval(str(show_images[0]))
            self.max_images = show_images[1]
        except NameError:
            self.show_images = False
            self.max_images = show_images[1]
        self.n_threads = n_threads

        self.raw_predictions = []
        self.predictions = {}

        # Define prediction storage paths
        self.raw_predictions_path = base_predictions_store_path+self.model_name+'/'+self.classification_name + \
            '/raw_predictions_.csv'
        self.predictions_paths_path = base_predictions_store_path+self.model_name+'/'+self.classification_name + \
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

        self.config = tf.ConfigProto(intra_op_parallelism_threads=self.n_threads)
        self.sess = tf.Session(config=self.config)

        self.model = Model(self.model_name, self.architecture, sess=self.sess)

        self.n_images = 0

    def build_dataset(self):
        self.csv_dataset_reader.read_file()
        self.X = self.csv_dataset_reader.X
        self.X_ = np.ndarray.tolist(self.X)
        self.n_images = len(self.X)

    def predict(self):
        self.sess = self.model.restore()

        x = tf.placeholder(tf.float32, [None, self.height, self.width, 3])

        convnet = model_map[self.architecture](self.height, self.width, self.n_classes)
        model = convnet.convnet(x)
        self.sess.run(tf.global_variables_initializer())

        self.X = self.sess.run(tf.reshape(self.X, [self.X.shape[0], self.height, self.width, 3]))
        predictions = self.sess.run(model, feed_dict={x: self.X})
        self.raw_predictions = np.ndarray.tolist(self.sess.run(tf.argmax(predictions, 1)))
        self._match_class_id()
    
    def write_predictions(self):
        mkdir(base_predictions_store_path)
        mkdir(base_predictions_store_path + '/' + self.model_name)
        mkdir(base_predictions_store_path + '/' + self.model_name + '/' + self.classification_name)

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


class LiveClassifier(TrainerMetadata):
    def __init__(self,
                 training_name: str,
                 n_threads: int):
        TrainerMetadata.__init__(self, training_name)

        self.training_name = training_name
        self.n_threads = n_threads

        self.config = tf.ConfigProto(intra_op_parallelism_threads=self.n_threads)
        self.sess = tf.Session(config=self.config)
        model = Model(self.model_name, self.architecture, sess=self.sess)
        self.sess = model.restore()

        self.x = tf.placeholder(tf.float32, [None, self.height, self.width, self.n_channels])
        self.convnet = model_map[self.architecture](self.height, self.width, self.n_classes)
        self.model = self.convnet.convnet(self.x)

        self.class_id_reader = JSONReader(None, workspace_dir + 'user/training_datasets/' +
                                          self.training_dataset_name + '/class_id.json')
        self.class_id_reader.bulk_read()
        self.class_id = self.class_id_reader.bulk_data

    def predict(self, images):
        if not isinstance(images, np.ndarray):
            raise ValueError("Images must be of type numpy.ndarray")
        if not images.shape[1:4] == (self.height, self.width, self.n_channels):
            raise AssertionError("Image is not of shape ", (self.height, self.width, self.n_channels))
        raw_predictions = self.sess.run(self.model, feed_dict={self.x: images})
        return self._match_class_id(raw_predictions)

    def reload_model(self):
        model = Model(self.model_name, self.architecture, sess=self.sess)
        self.sess = model.restore()

    def _match_class_id(self, raw_predictions):
        predictions = []
        for n, fp in enumerate(raw_predictions):
            predictions[n] = self.class_id[str(fp)]
        return tuple(predictions)
