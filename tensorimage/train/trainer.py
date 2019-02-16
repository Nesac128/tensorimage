import tensorflow as tf
import numpy as np
import warnings
import logging
import threading
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.exceptions import DataConversionWarning

from tensorimage.config.info import *
from tensorimage.file.reader import *
from tensorimage.file.writer import *
from tensorimage.data_augmentation._base import BaseOperation
from tensorimage.base.l2_regularization import L2RegularizationBuilder
from tensorimage.base.models.map.model import model_map
from tensorimage.base.model import Model, assert_exist
from tensorimage.base.metadata_reader import TrainingDatasetMetadata
import tensorimage.util.log as log

# Disable tensorflow & sklearn loggers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

sklearn_logger = log.getLogger('sklearn')
sklearn_logger.setLevel(logging.WARNING)
sklearn_logger.propagate = False


class Trainer(TrainingDatasetMetadata):
    def __init__(self,
                 data_name: str,
                 training_name: str,
                 n_epochs: int,
                 learning_rate: float,
                 l2_regularization_beta: float,
                 architecture: str,
                 data_augmentation_ops: tuple = (),
                 batch_size: int = 32,
                 train_test_split: float = 0.2,
                 n_threads: int = 10,
                 verbose=True):
        """
        :param data_name: unique name which identifies which image data to read for training
        :param n_epochs: number of epochs
        :param learning_rate: learning rate for optimizer
        :param l2_regularization_beta: value to use for L2 Regularization beta
        :param architecture: one of the CNN class architectures located in models/ directory
        :param data_augmentation_ops: data augmentation operation objects to apply to training data
        :param train_test_split: proportion of data that will be used as testing set
        """
        TrainingDatasetMetadata.__init__(self, data_name)

        self.data_name = data_name
        self.training_name = training_name
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.train_test_split = train_test_split
        self.l2_beta = l2_regularization_beta
        self.architecture = architecture
        self.data_augmentation_ops = data_augmentation_ops
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.verbose = verbose

        self.training_metadata_writer = JSONWriter(self.training_name, training_metafile_path)

        self.n_channels = 0
        self.model_name = self.training_name+'_model'

        self.csv_reader = CSVReader(self.training_data_path)

        self.X = None
        self.Y = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

        self.augmented_train_x = None
        self.augmented_train_y = None
        self.augmented_test_x = None
        self.augmented_test_y = None

        self.epochs = []
        self.training_accuracy = []
        self.training_cost = []
        self.testing_accuracy = []
        self.testing_cost = []

        self.config = tf.ConfigProto(intra_op_parallelism_threads=self.n_threads)
        self.sess = tf.Session(config=self.config)

        if assert_exist(self.model_name):
            self.model = Model(self.model_name, self.architecture, sess=self.sess)
            self.sess = self.model.restore()
        else:
            self.model = None

        self.base_op = None

    def build_dataset(self):
        log.info("Building dataset...", self)
        self.csv_reader.read_training_dataset(self.data_len, self.n_columns)
        self.X = self.csv_reader.X
        Y = self.csv_reader.Y

        encoder = LabelEncoder()
        encoder.fit(Y)
        y = encoder.transform(Y)
        self.Y = self._one_hot_encode(y)
        self.X, self.Y = shuffle(self.X, self.Y, random_state=1)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.X, self.Y,
                                                                                test_size=self.train_test_split,
                                                                                random_state=415)
        self.train_x = self.sess.run(tf.reshape(self.train_x, shape=[self.train_x.shape[0], self.height, self.width, 3]))
        self.test_x = self.sess.run(tf.reshape(self.test_x, shape=[self.test_x.shape[0], self.height, self.width, 3]))
        n_channels = self.train_x.shape[3]

        self.base_op = BaseOperation(self.train_x, self.train_y, self.n_classes, (self.height, self.width), n_channels)
        self.train_x, self.train_y = self.base_op.augment_images(*self.data_augmentation_ops)
        self.n_channels = self.train_x.shape[3]

    def train(self):
        x = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.train_x.shape[3]], name='x_'+self.training_name)
        labels = tf.placeholder(tf.float32, shape=[None, self.n_classes], name='labels_'+self.training_name)

        batch_iters = self.train_x.shape[0] // self.batch_size

        init = tf.global_variables_initializer()
        self.sess.run(init)

        convnet = model_map[self.architecture](self.height, self.width, self.n_channels, self.n_classes)
        model = convnet.convnet(x)

        l2_regularization_builder = L2RegularizationBuilder(self.architecture, self.l2_beta, (
            convnet.weights_shapes, convnet.biases_shapes))
        l2_regularization = l2_regularization_builder.start()

        with tf.name_scope('cost_function'):
            cost_function = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=model, labels=labels)) + l2_regularization)

        training_step = tf.train.AdamOptimizer(self.learning_rate, name='Adam_'+self.training_name).minimize(cost_function)

        self.sess.run(tf.global_variables_initializer())

        correct_prediction_ = tf.equal(tf.argmax(model, 1), tf.argmax(labels, 1))
        training_accuracy = tf.reduce_mean(tf.cast(correct_prediction_, tf.float32))
        testing_accuracy = tf.reduce_mean(tf.cast(correct_prediction_, tf.float32))

        with tf.name_scope('accuracy'):
            tf.summary.scalar('training_accuracy', training_accuracy)
            tf.summary.scalar('testing_accuracy', testing_accuracy)
        with tf.name_scope('cost'):
            tf.summary.scalar('training_cost', cost_function)

        log.info("Began training...", self)
        for epoch in range(1, self.n_epochs + 1):
            training_accuracy_ = 0
            testing_accuracy_ = 0
            training_cost = 0
            testing_cost = 0
            try:
                for i in range(1, batch_iters + 1):
                    x_, y = self._get_batch(i)
                    self.sess.run(training_step, feed_dict={x: x_, labels: y})

                    batch_training_accuracy = (self.sess.run(training_accuracy, feed_dict={x: x_, labels: y}))
                    batch_testing_accuracy = (
                        self.sess.run(testing_accuracy, feed_dict={x: self.test_x, labels: self.test_y}))
                    batch_testing_cost = self.sess.run(cost_function, feed_dict={x: self.test_x, labels: self.test_y})
                    batch_training_cost = self.sess.run(cost_function, feed_dict={x: x_, labels: y})

                    training_accuracy_ += batch_training_accuracy
                    testing_accuracy_ += batch_testing_accuracy
                    training_cost += batch_training_cost
                    testing_cost += batch_testing_cost
            except KeyboardInterrupt:
                break
            self.epochs.append(epoch)
            self.training_accuracy.append(training_accuracy_ / batch_iters)
            self.testing_accuracy.append(testing_accuracy_ / batch_iters)
            self.training_cost.append(training_cost)
            self.testing_cost.append(testing_cost)

        self._write_metadata()

    def _get_batch(self, i):
        return self.train_x[(self.batch_size*i)-self.batch_size:self.batch_size*i], \
                   self.train_y[(self.batch_size*i)-self.batch_size:self.batch_size*i]

    def _write_metadata(self):
        self.training_metadata_writer.update(data_name=self.data_name,
                                             model_name=self.model_name,
                                             dataset_name=self.dataset_name,
                                             architecture=self.architecture,
                                             n_classes=self.n_classes,
                                             height=self.height,
                                             width=self.width,
                                             n_channels=self.n_channels)
        self.training_metadata_writer.write()

    def store_model(self):
        if not self.model:
            model = Model(self.model_name, self.architecture, sess=self.sess)
            model.save()
        else:
            self.model.save()

    @staticmethod
    def _one_hot_encode(dlabels):
        n_labels = len(dlabels)
        n_unique_labels = len(np.unique(dlabels))
        one_hot_encode = np.zeros((n_labels, n_unique_labels))
        one_hot_encode[np.arange(n_labels), dlabels] = 1
        return one_hot_encode


class LiveTrainer:
    def __init__(self,
                 training_name: str,
                 architecture: str,
                 learning_rate: float,
                 n_epochs: int,
                 l2_regularization_beta: float,
                 height: int,
                 width: int,
                 n_classes: int,
                 n_channels: int,
                 n_threads: int = 10):
        self.training_name = training_name
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.l2_beta = l2_regularization_beta
        self.height = height
        self.width = width
        self.n_classes = n_classes
        self.n_threads = n_threads

        self.train_x = np.ndarray([0, height, width, n_channels])
        self.train_y = np.ndarray([0, n_classes])
        self.test_x = np.ndarray([0, height, width, n_channels])
        self.test_y = np.ndarray([0, n_classes])

        self.config = tf.ConfigProto(intra_op_parallelism_threads=self.n_threads)
        self.sess = tf.Session(config=self.config)

        self.model_name = self.training_name + '_model'
        if assert_exist(self.model_name):
            model = Model(self.model_name, self.architecture, sess=self.sess)
            self.sess = model.restore()

        self.x = tf.placeholder(tf.float32, shape=[None, height, width, n_channels])
        self.labels = tf.placeholder(tf.float32, shape=[None, self.n_classes])

        self.convnet = model_map[self.architecture](self.height, self.width, self.n_classes)
        self.model = self.convnet.convnet(self.x)

        self.epochs = []
        self.training_accuracy = []
        self.training_cost = []
        self.testing_accuracy = []
        self.testing_cost = []

        self.stop_ = False

    @staticmethod
    def threaded(fn):
        def wrapper(*args, **kwargs):
            threading.Thread(target=fn, args=args, kwargs=kwargs).start()
        return wrapper

    def add_test_data(self, test_x, test_y):
        if not isinstance(test_x, np.ndarray) or not isinstance(test_y, np.ndarray):
            raise ValueError("Testing data and labels must be of type numpy.ndarray")
        self.test_x = np.concatenate((self.test_x, test_x))
        self.test_y = np.concatenate((self.test_y, test_y))

    def add_training_data(self, train_x, train_y):
        if not isinstance(train_x, np.ndarray) or not isinstance(train_y, np.ndarray):
            raise ValueError("Training data and labels must be of type numpy.ndarray")
        self.train_x = np.concatenate((self.train_x, train_x))
        self.train_y = np.concatenate((self.train_y, train_y))

    def stop(self):
        self.stop_ = True

    @threaded
    def train(self):
        l2_regularization_builder = L2RegularizationBuilder(self.architecture, self.l2_beta, (
            self.convnet.weights_shapes, self.convnet.biases_shapes))
        l2_regularization = l2_regularization_builder.start()

        with tf.name_scope('cost_function'):
            cost_function = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.model, labels=self.labels)) + l2_regularization)

        training_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cost_function)

        correct_prediction_ = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.labels, 1))
        training_accuracy = tf.reduce_mean(tf.cast(correct_prediction_, tf.float32))
        testing_accuracy = tf.reduce_mean(tf.cast(correct_prediction_, tf.float32))

        while True:
            self.sess.run(training_step, feed_dict={self.x: self.train_x, self.labels: self.train_y})
            self.training_accuracy.append(self.sess.run(training_accuracy, feed_dict={self.x: self.train_x, self.labels: self.train_y}))
            self.testing_accuracy.append(self.sess.run(testing_accuracy, feed_dict={self.x: self.test_x, self.labels: self.test_y}))
            self.testing_cost.append(self.sess.run(cost_function, feed_dict={self.x: self.test_x, self.labels: self.test_y}))
            self.training_cost.append(self.sess.run(cost_function, feed_dict={self.x: self.train_x, self.labels: self.train_y}))
            if self.stop_:
                break

    def update_model(self):
        model = Model(self.model_name, self.architecture, sess=self.sess)
        model.save()
