import tensorflow as tf
import numpy as np
import csv
import time
import datetime

from src.config import *
from src.neural_network_models.convolutional.convolutional_neural_network import convolutional_neural_network
from src.image.display import display_image
from src.man.reader import *
from src.man.mkdir import mkdir


class Predict:
    def __init__(self,
                 id_name,
                 model_folder_name,
                 model_name,
                 training_dataset_name,
                 prediction_dataset_name,
                 show_image: bool=True):
        self.id_name = id_name
        self.model_folder_name = model_folder_name
        self.model_name = model_name
        self.training_dataset_name = training_dataset_name
        self.prediction_dataset_name = prediction_dataset_name
        try:
            self.show_image = eval(str(show_image))
        except NameError:
            self.show_image = False

        self.raw_predictions = []
        self.predictions = []

        self.id_name_metadata_reader = JSONReader(self.id_name, nid_names_metafile_path)
        self.id_name_metadata_reader.bulk_read()
        self.id_name_metadata_reader.select()
        self.data_id = self.id_name_metadata_reader.selected_data["id"]
        self.metadata_reader = JSONReader(str(self.data_id), dataset_metafile_path)
        self.metadata_reader.bulk_read()
        self.metadata_reader.select()

        image_metadata = self.metadata_reader.selected_data
        self.data_path = image_metadata["data_path"]
        self.width = image_metadata["width"]
        self.height = image_metadata["height"]
        self.path_file = image_metadata["path_file"]

        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S').replace('-', '_')
        timestamp = timestamp.replace(' ', '_')
        timestamp = timestamp.replace(':', '_')

        predictions_path = external_working_directory_path+'user/predictions/'+self.prediction_dataset_name
        mkdir(predictions_path)
        mkdir(predictions_path+'/'+self.model_folder_name)
        self.prediction_filenames = \
            [external_working_directory_path+'user/predictions/'+self.prediction_dataset_name + '/' +
                self.model_folder_name + '/raw_predictions_'+timestamp+'.csv',
                external_working_directory_path+'user/predictions/'+self.prediction_dataset_name + '/' +
                self.model_folder_name + '/predictions_to_paths'+timestamp+'.csv']

        self.csv_dataset_reader = CSVReader(self.data_path)
        self.class_id_reader = JSONReader(None, external_working_directory_path+'user/datasets/' +
                                          self.training_dataset_name + '/class_id.json')

        self.path_reader = TXTReader(self.path_file)
        self.path_reader.read_raw()
        self.path_reader.parse()
        self.paths = self.path_reader.parsed_data

        self.class_id_reader.bulk_read()
        self.class_id = self.class_id_reader.bulk_data

        self.model_restore = RestoreModel(self.model_folder_name, self.model_name)

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
        self.read_dataset()
        self.model_restore.restore_cnn_model1_params(sess)
        self.weights, self.biases = self.model_restore.weights, self.model_restore.biases

        x = tf.placeholder(tf.float32, [None, self.height, self.width, 3])

        model, conv1 = convolutional_neural_network(x, self.weights, self.biases)

        self.read_dataset()
        self.X = sess.run(tf.reshape(self.X, [self.X.shape[0], self.height, self.width, 3]))
        prediction = sess.run(model, feed_dict={x: self.X})
        self.raw_predictions = np.ndarray.tolist(sess.run(tf.argmax(prediction, 1)))

    def match_class_id(self):
        for rp in self.raw_predictions:
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
        for image_n in range(len(self.paths)):
            if self.show_image:
                display_image(self.predictions[image_n], self.paths[image_n])


class RestoreModel:
    def __init__(self, model_folder_name, model_name):
        self.model_folder_name = model_folder_name
        self.model_name = model_name

        self.weights = {}
        self.biases = {}

    def restore_cnn_model1_params(self, sess):
        saver = tf.train.import_meta_graph(external_working_directory_path + 'user/trained_models/' +
                                           self.model_folder_name + '/' + self.model_name + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(external_working_directory_path + 'user/trained_models/' +
                                                       self.model_folder_name + './'))
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
