import tensorflow as tf
import numpy as np
import csv
import time
import datetime

from src.config import *
from src.neural_network_models.convolutional.cnn_model1 import convolutional_neural_network
from src.image.display import display_image
from src.man.reader import *
from src.man.mkdir import mkdir


class Predict:
    def __init__(self,
                 data_name,
                 training_name,
                 classification_name,
                 show_image: bool=False):
        self.data_name = data_name
        self.training_name = training_name
        self.classification_name = classification_name
        try:
            self.show_image = eval(str(show_image))
        except NameError:
            self.show_image = False

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
        self.prediction_dataset_name = image_metadata["dataset_name"]

        # Read trained model metadata
        self.training_metadata_reader = JSONReader(self.training_name, training_metafile_path)
        self.training_metadata_reader.bulk_read()
        self.training_metadata_reader.select()
        training_metadata = self.training_metadata_reader.selected_data
        self.model_path = training_metadata["model_folder_name"]
        self.model_name = training_metadata["model_name"]
        self.cnn_architecture = training_metadata["cnn_architecture"]
        self.training_dataset_name = training_metadata["dataset_name"]
        self.class_scores = training_metadata["class_scores"]
        self.model_folder_name = self.model_path.split('/')[-1]

        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S').replace('-', '_')
        timestamp = timestamp.replace(' ', '_')
        timestamp = timestamp.replace(':', '_')
        predictions_path = workspace_dir+'user/predictions/'+self.prediction_dataset_name
        mkdir(predictions_path)
        mkdir(predictions_path+'/'+self.model_folder_name)
        self.prediction_filenames = \
            [workspace_dir+'user/predictions/'+self.prediction_dataset_name + '/' +
                self.model_folder_name + '/raw_predictions_'+timestamp+'.csv',
                workspace_dir+'user/predictions/'+self.prediction_dataset_name + '/' +
                self.model_folder_name + '/predictions_to_paths'+timestamp+'.csv']

        self.csv_dataset_reader = CSVReader(self.data_path)
        self.class_id_reader = JSONReader(None, workspace_dir+'user/training_datasets/' +
                                          self.training_dataset_name + '/class_id.json')

        self.path_reader = TXTReader(self.path_file)
        self.path_reader.read_raw()
        self.path_reader.parse()
        self.paths = self.path_reader.parsed_data

        self.class_id_reader.bulk_read()
        self.class_id = self.class_id_reader.bulk_data

        self.model_restore = RestoreModel(self.model_path, self.model_name)

        self.X = None
        self.weights = None
        self.biases = None

        self.list_X = None

    def read_dataset(self):
        self.csv_dataset_reader.read_file()
        self.X = self.csv_dataset_reader.X

        self.list_X = np.ndarray.tolist(self.X)

    def predict(self):
        sess = tf.Session()
        with sess.as_default():
            self.read_dataset()
            self.model_restore.restore_cnn_model1_params(sess)
            self.weights, self.biases = self.model_restore.weights, self.model_restore.biases

            x = tf.placeholder(tf.float32, [None, self.height, self.width, 3])

            model = convolutional_neural_network(x, self.weights, self.biases)

            self.read_dataset()
            self.X = sess.run(tf.reshape(self.X, [self.X.shape[0], self.height, self.width, 3]))
            predictions = model.eval(feed_dict={x: self.X})
            self.final_predictions = np.ndarray.tolist(sess.run(tf.argmax(predictions, 1)))

    def match_class_id(self):
        for rp in self.final_predictions:
            self.predictions.append(self.class_id[str(rp)])

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
            for image_n in range(len(self.paths)):
                writer.writerow([self.paths[image_n], self.predictions[image_n]])
        if self.show_image:
            for image_n in range(len(self.paths)):
                display_image(self.predictions[image_n],
                              self.paths[image_n])


class RestoreModel:
    def __init__(self, model_folder_name, model_name):
        self.model_folder_name = model_folder_name
        self.model_name = model_name

        self.weights = {}
        self.biases = {}

    def restore_cnn_model1_params(self, sess):
        saver = tf.train.import_meta_graph(workspace_dir + 'user/trained_models/' +
                                           self.model_folder_name + '/' + self.model_name + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(workspace_dir + 'user/trained_models/' +
                                                       self.model_folder_name + '/./'))
        with tf.name_scope('weights'):
            self.weights = {
                'conv1': sess.run('weights/conv1:0'),
                'conv2': sess.run('weights/conv2:0'),
                'fcl': sess.run('weights/fcl:0'),
                'out': sess.run('weights/out:0')
            }
        with tf.name_scope('biases'):
            self.biases = {
                'conv1': sess.run('biases/conv1:0'),
                'conv2': sess.run('biases/conv2:0'),
                'fcl': sess.run('biases/fcl:0'),
                'out': sess.run('biases/out:0')
            }

        for w in self.weights:
            self.weights[w] = tf.convert_to_tensor(self.weights[w])
        for b in self.biases:
            self.biases[b] = tf.convert_to_tensor(self.biases[b])
