import tensorflow as tf
import numpy as np
import csv

from tensorimage.tensorimage.config import *
from tensorimage.tensorimage.src.image.display import display_image
from tensorimage.tensorimage.src.file.reader import *
from tensorimage.tensorimage.src.os.mkdir import mkdir
from tensorimage.tensorimage.src.convnet_builder import ConvNetBuilder
from tensorimage.tensorimage.src.classify.restore_model import ModelRestorer


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
        identify the model to use for carrying out the predictions
        :param classification_name: name that will be used to organize the output predictions in your workspace
        :param show_images: boolean specifying on whether to display all of the images with class predictions
        """
        self.data_name = data_name
        self.training_name = training_name
        self.classification_name = classification_name
        try:
            self.show_images = eval(str(show_images[0]))
            self.max_images = show_images[1]
        except NameError:
            self.show_images = False
            self.max_images = show_images[1]

        self.predictions = []
        self.final_predictions = []

        # Read unclassified data metadata
        self.metadata_reader = JSONReader(self.data_name, dataset_metafile_path)
        self.metadata_reader.bulk_read()
        self.metadata_reader.select()
        image_metadata = self.metadata_reader.selected_data
        self.data_path = image_metadata["data_path"]
        self.width = image_metadata["width"]
        self.height = image_metadata["height"]
        self.path_file = image_metadata["path_file"]
        self.n_images = image_metadata["n_images"]
        self.n_classes = image_metadata["n_classes"]

        # Read trained model metadata
        self.training_metadata_reader = JSONReader(self.training_name, training_metafile_path)
        self.training_metadata_reader.bulk_read()
        self.training_metadata_reader.select()
        training_metadata = self.training_metadata_reader.selected_data
        self.model_path = training_metadata["model_folder_name"]
        self.model_name = training_metadata["model_name"]
        self.cnn_architecture = training_metadata["cnn_architecture"]
        self.training_dataset_name = training_metadata["dataset_name"]
        self.model_folder_name = self.model_path.split('/')[-1]
        self.architecture = training_metadata["architecture"]

        predictions_store_path = base_predictions_store_path+self.model_folder_name
        mkdir(predictions_store_path)
        mkdir(predictions_store_path+'/'+self.model_folder_name)
        self.prediction_filenames = \
            [workspace_dir+'user/predictions/'+self.model_folder_name + '/' +
                self.classification_name + '/raw_predictions_.csv',
                workspace_dir+'user/predictions/'+self.model_folder_name + '/' +
                self.classification_name + '/predictions_to_paths.csv']

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
        self.weights = None
        self.biases = None
        self.list_X = None

        self.convnet_builder = ConvNetBuilder(self.architecture)
        self.convolutional_neural_network = self.convnet_builder.build_convnet()

        self.sess = tf.Session()
        self.model_restore = ModelRestorer(self.model_path, self.model_name, self.architecture, self.sess)

    def build_dataset(self):
        self.csv_dataset_reader.read_file()
        self.X = self.csv_dataset_reader.X

        self.list_X = np.ndarray.tolist(self.X)

    def predict(self):
        with self.sess.as_default():
            self.model_restore.start()

            x = tf.placeholder(tf.float32, [None, self.height, self.width, 3])

            convnet = self.convolutional_neural_network(x, self.n_classes)
            model = convnet.convnet()

            self.X = self.sess.run(tf.reshape(self.X, [self.X.shape[0], self.height, self.width, 3]))
            predictions = model.eval(feed_dict={x: self.X})
            self.final_predictions = np.ndarray.tolist(self.sess.run(tf.argmax(predictions, 1)))
            self._match_class_id()
    
    def write_predictions(self):
        # Write image data in new file with predicted classes
        for image_n in range(len(self.list_X)):
            self.list_X[image_n].append(self.predictions[image_n])

        with open(self.prediction_filenames[0], 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for image_prediction in self.list_X:
                writer.writerow(image_prediction)

        with open(self.prediction_filenames[1], 'a') as pathfile:
            writer = csv.writer(pathfile, delimiter=',')
            for n in range(self.n_images):
                writer.writerow([self.image_paths[n], self.predictions[n]])
        if self.show_images:
            for n in zip(range(self.n_images), range(self.max_images)):
                display_image(self.predictions[n[0]],
                              self.image_paths[n[0]])

    def _match_class_id(self):
        for rp in self.final_predictions:
            self.predictions.append(self.class_id[str(rp)])
