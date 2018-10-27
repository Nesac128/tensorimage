import tensorflow as tf
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from src.config import *
from src.exceptions import *
from src.man.reader import *
from src.man.writer import *


class Train:
    def __init__(self,
                 data_name: str,
                 training_name: str,
                 n_epochs,
                 learning_rate,
                 l2_regularization_beta,
                 batch_size: int = 32,
                 train_test_split: float = 0.2,
                 augment_data=False,
                 cnn_architecture='cnn_model1'):
        """
        :param data_name: unique name which identifies which image data to read for training
        :param n_epochs: number of epochs
        :param learning_rate: learning rate for optimizer
        :param l2_regularization_beta: value to use for L2 Regularization beta
        :param train_test_split: proportion of data that will be used as testing set
        :param cnn_architecture: cnn architecture name (e.g: alexnet)
        """
        # Store parameter inputs
        self.data_name = data_name
        self.training_name = training_name
        self.n_epochs = int(n_epochs)
        self.learning_rate = float(learning_rate)
        self.train_test_split = float(train_test_split)
        self.l2_beta = float(l2_regularization_beta)
        self.batch_size = int(batch_size)
        try:
            self.augment_data = eval(str(augment_data))
        except NameError:
            self.augment_data = False
        self.cnn_architecture = cnn_architecture

        self.training_metadata_writer = JSONWriter(self.training_name, training_metafile_path)

        self.metadata_reader = JSONReader(self.data_name, dataset_metafile_path)
        self.metadata_reader.bulk_read()
        self.metadata_reader.select()
        image_metadata = self.metadata_reader.selected_data
        self.n_columns = image_metadata["n_columns"]
        self.training_data_path = image_metadata["data_path"]
        self.n_classes = image_metadata["n_classes"]
        self.dataset_type = image_metadata["dataset_type"]
        self.is_trainable()
        self.data_len = image_metadata["data_len"]
        self.width = image_metadata["width"]
        self.height = image_metadata["height"]
        self.dataset_name = image_metadata["dataset_name"]

        self.model_folder_name = self.training_name+'_model'

        self.csv_reader = CSVReader(self.training_data_path)

        self.X = None
        self.Y = None
        self.train_x = None
        self.train_y = None

        self.bcn = BuildConvNet(self.cnn_architecture, self.n_classes, self.l2_beta)
        self.bcn.build_convnet()
        self.weights, self.biases = self.bcn.weights, self.bcn.biases
        self.convolutional_neural_network = self.bcn.cnn_model
        self.bcn.build_l2_reg()
        self.l2_reg = self.bcn.l2_reg

        self.build_dataset()

    def build_dataset(self):
        self.csv_reader.read_training_dataset(self.data_len, self.n_columns)
        self.X = self.csv_reader.X
        Y = self.csv_reader.Y

        print(self.X.shape, "X shape")
        print(Y.shape, "Y shape")

        encoder = LabelEncoder()
        encoder.fit(Y)
        y = encoder.transform(Y)
        self.Y = self.one_hot_encode(y)

    def one_hot_encode(self, dlabels):
        n_labels = len(dlabels)
        n_unique_labels = len(np.unique(dlabels))
        one_hot_encode = np.zeros((n_labels, n_unique_labels))
        one_hot_encode[np.arange(n_labels), dlabels] = 1
        return one_hot_encode

    def get_next_batch(self, i):
        return self.train_x[(self.batch_size*i)-self.batch_size:self.batch_size*i], \
                   self.train_y[(self.batch_size*i)-self.batch_size:self.batch_size*i]

    def train_convolutional(self):
        sess = tf.Session()

        if self.augment_data:
            self.augment_data_()

        X, Y = shuffle(self.X, self.Y, random_state=1)
        train_x, test_x, self.train_y, test_y = train_test_split(X, Y, test_size=self.train_test_split, random_state=415)
        self.train_x = sess.run(tf.reshape(train_x, shape=[train_x.shape[0], self.height, self.width, 3]))
        test_x = sess.run(tf.reshape(test_x, shape=[test_x.shape[0], self.height, self.width, 3]))

        x = tf.placeholder(tf.float32, [None, self.height, self.width, 3], name='x')
        labels = tf.placeholder(tf.float32, [None, self.n_classes])

        batch_iters = train_x.shape[0] // self.batch_size

        init = tf.global_variables_initializer()
        sess.run(init)

        os.system("clear")

        model = self.convolutional_neural_network(x, self.weights, self.biases)

        with tf.name_scope('cost_function'):
            cost_function = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=model, labels=labels)) + self.l2_reg)

        print("Optimizer: Adam")
        training_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cost_function)

        sess.run(tf.global_variables_initializer())

        correct_prediction_ = tf.equal(tf.argmax(model, 1), tf.argmax(labels, 1))
        training_accuracy = tf.reduce_mean(tf.cast(correct_prediction_, tf.float32))
        testing_accuracy = tf.reduce_mean(tf.cast(correct_prediction_, tf.float32))

        with tf.name_scope('accuracy'):
            tf.summary.scalar('training_accuracy', training_accuracy)
            tf.summary.scalar('testing_accuracy', testing_accuracy)
        with tf.name_scope('cost'):
            tf.summary.scalar('training_cost', cost_function)

        write_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(workspace_dir+'user/logs/'+str(self.training_name), sess.graph)
        for epoch in range(self.n_epochs):
            estop = False
            training_accuracy_ = 0
            testing_accuracy_ = 0
            training_cost = 0
            testing_cost = 0
            for i in range(batch_iters):
                x_, y = self.get_next_batch(i+1)
                try:
                    sess.run(training_step, feed_dict={x: x_, labels: y})

                    batch_training_accuracy = (sess.run(training_accuracy, feed_dict={x: x_, labels: y}))
                    batch_testing_accuracy = (sess.run(testing_accuracy, feed_dict={x: test_x, labels: test_y}))
                    batch_testing_cost = sess.run(cost_function, feed_dict={x: test_x, labels: test_y})
                    batch_training_cost = sess.run(cost_function, feed_dict={x: x_, labels: y})

                    training_accuracy_ += batch_training_accuracy
                    testing_accuracy_ += batch_testing_accuracy
                    training_cost += batch_training_cost
                    testing_cost += batch_testing_cost

                except KeyboardInterrupt:
                    estop = self.early_stop()
                    if estop:
                        break
                    else:
                        continue
            if not estop:
                avr_training_accuracy = training_accuracy_/batch_iters
                avr_testing_accuracy = testing_accuracy_/batch_iters

                summary = sess.run(write_op, {training_accuracy: avr_training_accuracy,
                                              testing_accuracy: avr_testing_accuracy,
                                              cost_function: training_cost})
                writer.add_summary(summary, epoch)
                writer.flush()
                print("Epoch ", epoch + 1, "   Training accuracy: ", float("%0.5f" % avr_training_accuracy),
                      "   Testing accuracy: ", float("%0.5f" % avr_testing_accuracy), "   Training cost ",
                      training_cost, "   Testing cost: ", testing_cost)
            else:
                break

        self.store_model(sess)
        writer.close()

    def write_metadata(self):
        self.training_metadata_writer.update(data_name=self.data_name,
                                             model_folder_name=self.model_folder_name,
                                             model_name=self.training_name,
                                             cnn_architecture=self.cnn_architecture,
                                             dataset_name=self.dataset_name)
        self.training_metadata_writer.write()

    def augment_data_(self):
        augmented_data = tf.constant([], tf.float32, shape=[0, self.height, self.width, 3])
        augmented_labels = tf.constant([], tf.float32, shape=[0, self.n_classes])
        for n_image, n_label in zip(range(self.train_x.shape[0]), range(self.train_y.shape[0])):
            flip_1 = tf.image.flip_up_down(self.train_x[n_image])
            flip_2 = tf.image.flip_left_right(self.train_x[n_image])
            flip_3 = tf.image.random_flip_up_down(self.train_x[n_image])
            flip_4 = tf.image.random_flip_left_right(self.train_x[n_image])

            pre_data = tf.concat([tf.expand_dims(self.train_x[n_image], 0), tf.expand_dims(flip_1, 0),
                                  tf.expand_dims(flip_2, 0), tf.expand_dims(flip_3, 0),
                                  tf.expand_dims(flip_4, 0)], 0)

            pre_labels = tf.concat([tf.expand_dims(self.train_y[n_label], 0), tf.expand_dims(self.train_y[n_label], 0),
                                    tf.expand_dims(self.train_y[n_label], 0), tf.expand_dims(self.train_y[n_label], 0),
                                    tf.expand_dims(self.train_y[n_label], 0)], 0)
            augmented_data = tf.concat([augmented_data, tf.cast(pre_data, tf.float32)], 0)
            augmented_labels = tf.concat([augmented_labels, tf.cast(pre_labels, tf.float32)], 0)

        return augmented_data, augmented_labels

    def is_trainable(self):
        if self.dataset_type != 'training':
            raise DataNotTrainableError('Data is not trainable: {}'.format(self.training_data_path))

    def store_model(self, sess):
        saver = tf.train.Saver()
        saver.save(sess, workspace_dir + 'user/trained_models/' + self.model_folder_name
                   + '/' + self.training_name)

    def early_stop(self):
        early_stop = input("You have chosen to early stop the training process. Do you wish to proceed? [Y/n]")
        if early_stop == "Y" or early_stop == "y":
            print("Saving model...")
            return True
        else:
            print("Resuming training...")
            return False


class BuildWeightsBiases:
    def __init__(self, n_classes):
        self.n_classes = n_classes

        self.weights = {}
        self.biases = {}
        self.cnn_model = None

    def build_cnn_model1_params(self):
        from src.neural_network_models.convolutional.cnn_model1 import convolutional_neural_network
        self.cnn_model = convolutional_neural_network

        with tf.name_scope('weights'):
            self.weights = {
                'conv1': tf.Variable(tf.random_normal([5, 5, 3, 3]), name='conv1'),
                'conv2': tf.Variable(tf.random_normal([3, 3, 3, 64]), name='conv2'),
                'fcl': tf.Variable(tf.random_normal([7*7*64, 128]), name='fcl'),
                'out': tf.Variable(tf.random_normal([128, self.n_classes]), name='out')
            }
        with tf.name_scope('biases'):
            self.biases = {
                'conv1': tf.Variable(tf.random_normal([3]), name='conv1'),
                'conv2': tf.Variable(tf.random_normal([64]), name='conv2'),
                'fcl': tf.Variable(tf.random_normal([128]), name='fcl'),
                'out': tf.Variable(tf.random_normal([self.n_classes]), name='out')
            }

    def build_alexnet_params(self):
        from src.neural_network_models.convolutional.alexnet import convolutional_neural_network
        self.cnn_model = convolutional_neural_network

        with tf.name_scope('weights'):
            self.weights = {
                'conv1': tf.Variable(tf.random_normal([11, 11, 3, 96]), name='conv1'),
                'conv2': tf.Variable(tf.random_normal([5, 5, 96, 256]), name='conv2'),
                'conv3': tf.Variable(tf.random_normal([3, 3, 256, 384]), name='conv3'),
                'conv4': tf.Variable(tf.random_normal([3, 3, 384, 384]), name='conv4'),
                'conv5': tf.Variable(tf.random_normal([3, 3, 384, 256]), name='conv5'),
                'fcl': tf.Variable(tf.random_normal([8 * 8 * 256, 4096]), name='fcl'),
                'fcl2': tf.Variable(tf.random_normal([4096, 4096]), name='fcl2'),
                'out': tf.Variable(tf.random_normal([4096, self.n_classes]), name='out')
            }
        with tf.name_scope('biases'):
            self.biases = {
                'conv1': tf.Variable(tf.random_normal([96]), name='conv1'),
                'conv2': tf.Variable(tf.random_normal([256]), name='conv2'),
                'conv3': tf.Variable(tf.random_normal([384]), name='conv3'),
                'conv4': tf.Variable(tf.random_normal([384]), name='conv4'),
                'conv5': tf.Variable(tf.random_normal([256]), name='conv5'),
                'fcl': tf.Variable(tf.random_normal([4096]), name='fcl'),
                'fcl2': tf.Variable(tf.random_normal([4096]), name='fcl2'),
                'out': tf.Variable(tf.random_normal([self.n_classes]), name='out')
            }


class BuildConvNet(BuildWeightsBiases):
    def __init__(self, cnn_architecture, n_classes, l2_beta):
        self.cnn_architecture = cnn_architecture
        self.n_classes = n_classes
        self.l2_beta = l2_beta

        BuildWeightsBiases.__init__(self, n_classes)

        self.l2_reg = None

    def build_convnet(self):
        if self.cnn_architecture == "cnn_model1":
            self.build_cnn_model1_params()
        elif self.cnn_architecture == "alexnet":
            self.build_alexnet_params()

    def build_l2_reg(self):
        if self.cnn_architecture == "cnn_model1":
            self.l2_reg = (self.l2_beta * tf.nn.l2_loss(self.weights['conv1']) +
                           self.l2_beta * tf.nn.l2_loss(self.biases['conv1']) +
                           self.l2_beta * tf.nn.l2_loss(self.weights['conv2']) +
                           self.l2_beta * tf.nn.l2_loss(self.biases['conv2']) +
                           self.l2_beta * tf.nn.l2_loss(self.weights['fcl']) +
                           self.l2_beta * tf.nn.l2_loss(self.biases['fcl']) +
                           self.l2_beta * tf.nn.l2_loss(self.weights['out']) +
                           self.l2_beta * tf.nn.l2_loss(self.biases['out']))
        elif self.cnn_architecture == "alexnet":
            self.l2_reg = (self.l2_beta * tf.nn.l2_loss(self.weights['conv1']) +
                           self.l2_beta * tf.nn.l2_loss(self.biases['conv1']) +
                           self.l2_beta * tf.nn.l2_loss(self.weights['conv2']) +
                           self.l2_beta * tf.nn.l2_loss(self.biases['conv2']) +
                           self.l2_beta * tf.nn.l2_loss(self.weights['conv3']) +
                           self.l2_beta * tf.nn.l2_loss(self.biases['conv3']) +
                           self.l2_beta * tf.nn.l2_loss(self.weights['conv4']) +
                           self.l2_beta * tf.nn.l2_loss(self.biases['conv4']) +
                           self.l2_beta * tf.nn.l2_loss(self.weights['conv5']) +
                           self.l2_beta * tf.nn.l2_loss(self.biases['conv5']) +
                           self.l2_beta * tf.nn.l2_loss(self.weights['fcl']) +
                           self.l2_beta * tf.nn.l2_loss(self.biases['fcl2']) +
                           self.l2_beta * tf.nn.l2_loss(self.weights['out']) +
                           self.l2_beta * tf.nn.l2_loss(self.biases['out']))
