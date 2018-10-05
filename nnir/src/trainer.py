import tensorflow as tf
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from matplotlib import style
import matplotlib.pyplot as plt
import pandas as pd
import time

from src.pcontrol import *
from src.config import *
from src.exceptions import *
from src.man.reader import *
from src.man.writer import *
from src.neural_network_models.convolutional.convolutional_neural_network import convolutional_neural_network
from src.meta.id import ID
from live_data_streaming import server


class Train:
    def __init__(self,
                 data_id,
                 model_store_path,
                 model_name,
                 optimizer='GradientDescent',
                 display=True,
                 display_frequency: int=50,
                 n_epochs: int = 1000,
                 learning_rate: float = 0.2,
                 train_test_split: float = 0.3,
                 l2_regularization_beta=0.01):
        self.labels = []

        # Store parameter inputs
        self.data_id = data_id
        self.model_store_path = model_store_path
        self.model_name = model_name
        self.optimizer = optimizer
        self.display = display
        self.display_frequency = display_frequency
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.train_test_split = train_test_split
        self.l2_reg_beta = l2_regularization_beta

        # Define necessary Objects
        self.id_man = ID('training')
        self.id_man.read()
        self.training_id = self.id_man.id

        self.id_man = ID('accuracy')
        self.id_man.read()
        self.accuracy_id = self.id_man.id

        self.metadata_writer = JSONWriter(self.training_id, training_metafile_path)
        self.accuracy_writer = JSONWriter(self.accuracy_id, accuracy_metafile_path)

        self.metadata_reader = JSONReader(self.data_id, dataset_metafile_path)
        self.metadata_reader.bulk_read()
        self.metadata_reader.select()

        # Store metadata in variables
        self.n_columns = self.metadata_reader.selected_data["n_columns"]
        self.training_data_path = self.metadata_reader.selected_data["data_path"]
        self.n_classes = self.metadata_reader.selected_data["n_classes"]
        self.trainable = self.metadata_reader.selected_data["trainable"]
        self.data_len = self.metadata_reader.selected_data["data_len"]
        self.width = self.metadata_reader.selected_data["width"]
        self.height = self.metadata_reader.selected_data["height"]

        self.csv_reader = CSVReader(self.training_data_path)
        self.check_data_type()

        self.data_streaming_server = server.LiveDataStreamingServer(2025)

        self.plot_epoch = []
        self.testing_mse_history = []
        self.training_mse_history = []
        self.training_cost_history = []
        self.testing_cost_history = []
        self.training_accuracy_history = []
        self.testing_accuracy_history = []

        self.X = None
        self.Y = None
        self.weights = None
        self.biases = None

    def read_dataset(self):
        self.csv_reader.read_training_dataset(self.data_len, self.n_columns)
        self.X = self.csv_reader.X
        Y = self.csv_reader.Y

        print(self.X.shape, "X shape")
        print(Y.shape, "Y shape")

        global labels
        labels = pd.Series.tolist(Y)

        encoder = LabelEncoder()
        encoder.fit(Y)
        y = encoder.transform(Y)
        self.Y = self.one_hot_encoder(y)

    def one_hot_encoder(self, dlabels):
        n_labels = len(dlabels)
        n_unique_labels = len(np.unique(dlabels))
        one_hot_encode = np.zeros((n_labels, n_unique_labels))
        one_hot_encode[np.arange(n_labels), dlabels] = 1
        return one_hot_encode

    def gen_weights_biases(self):
        # self.weights = {'conv_weights1': tf.Variable(tf.random_normal([5, 5, 3, 30]), name='conv_weights1'),
        #                 'conv_weights2': tf.Variable(tf.random_normal([3, 3, 30, 15]), name='conv_weights2'),
        #                 'fcl_weights3': tf.Variable(tf.random_normal([2*2*15, 128]), name='fcl_weights3'),
        #                 'out_weights4': tf.Variable(tf.random_normal([128, self.n_classes]), name='out_weights4')
        #                 }
        # self.biases = {'conv_biases1': tf.Variable(tf.random_normal([30]), name='conv_biases1'),
        #                'conv_biases2': tf.Variable(tf.random_normal([15]), name='conv_biases2'),
        #                'fcl_biases3': tf.Variable(tf.random_normal([128]), name='fcl_biases3'),
        #                'out_biases4': tf.Variable(tf.random_normal([self.n_classes]), name='out_biases4')
        #                }
        self.weights = {'conv_weights1': tf.Variable(tf.random_normal([5, 5, 3, 32]), name='conv_weights1'),
                        'conv_weights2': tf.Variable(tf.random_normal([3, 3, 32, 64]), name='conv_weights2'),
                        # 'conv_weights3': tf.Variable(tf.random_normal([3, 3, 64, 128])),
                        'fcl_weights3': tf.Variable(tf.random_normal([2*2*64, 128]), name='fcl_weights3'),
                        'out_weights4': tf.Variable(tf.random_normal([128, self.n_classes]), name='out_weights4')
                        }
        self.biases = {'conv_biases1': tf.Variable(tf.random_normal([32]), name='conv_biases1'),
                       'conv_biases2': tf.Variable(tf.random_normal([64]), name='conv_biases2'),
                       'fcl_biases3': tf.Variable(tf.random_normal([128]), name='fcl_biases3'),
                       'out_biases4': tf.Variable(tf.random_normal([self.n_classes]), name='out_biases4')
                       }

    def train_convolutional(self):
        sess = tf.Session()
        print("Reading dataset...")
        self.read_dataset()

        X, Y = shuffle(self.X, self.Y, random_state=1)
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=self.train_test_split, random_state=415)
        train_x = sess.run(tf.reshape(train_x, shape=[train_x.shape[0], self.height, self.width, 3]))
        test_x = sess.run(tf.reshape(test_x, shape=[test_x.shape[0], self.height, self.width, 3]))

        # train_x, train_y = sess.run(self.augment_data(train_x, train_y))
        # test_x, test_y = sess.run(self.augment_data(test_x, test_y))

        x = tf.placeholder(tf.float32, [None, self.height, self.width, 3], name='x')
        print(x.shape)

        sess.run(tf.global_variables_initializer())
        os.system("clear")
        self.gen_weights_biases()
        model = convolutional_neural_network(x, self.weights, self.biases)

        labels = tf.placeholder(tf.float32, [None, self.n_classes])

        cost_function = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=model, labels=labels)) +
                         self.l2_reg_beta * tf.nn.l2_loss(self.weights['conv_weights1']) +
                         self.l2_reg_beta * tf.nn.l2_loss(self.biases['conv_biases1']) +
                         self.l2_reg_beta * tf.nn.l2_loss(self.weights['conv_weights2']) +
                         self.l2_reg_beta * tf.nn.l2_loss(self.biases['conv_biases2']) +
                         self.l2_reg_beta * tf.nn.l2_loss(self.weights['fcl_weights3']) +
                         self.l2_reg_beta * tf.nn.l2_loss(self.biases['fcl_biases3']) +
                         self.l2_reg_beta * tf.nn.l2_loss(self.weights['out_weights4']) +
                         self.l2_reg_beta * tf.nn.l2_loss(self.biases['out_biases4']))

        training_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost_function)

        sess.run(tf.global_variables_initializer())

        print("Began setting up data streaming server!")
        self.data_streaming_server.main()
        print("Finished setting up data streaming server!")

        print("Began training... ")
        for epoch in range(self.n_epochs):
            # if epoch != 0:
            #     if epoch % self.display_frequency == 0:
            #         self.display_progress()
            #         os.system("clear")
            # self.plot_epoch.append(epoch)
            starting_time = time.time()

            sess.run(training_step, feed_dict={x: train_x, labels: train_y})
            correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(labels, 1))

            training_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            training_accuracy = (sess.run(training_accuracy, feed_dict={x: train_x, labels: train_y}))

            testing_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            testing_accuracy = (sess.run(testing_accuracy, feed_dict={x: test_x, labels: test_y}))

            test_y_prediction = sess.run(model, feed_dict={x: test_x})
            testing_mse = sess.run(tf.reduce_mean(tf.square(test_y_prediction - test_y)))

            # training_y_prediction = sess.run(model, feed_dict={x: train_x})
            # training_mse = sess.run(tf.reduce_mean(tf.square(training_y_prediction - train_x)))

            training_cost = sess.run(cost_function, feed_dict={x: train_x, labels: train_y})
            testing_cost = sess.run(cost_function, feed_dict={x: test_x, labels: test_y})

            server.update_data(epoch=epoch,
                               training_accuracy=training_accuracy,
                               training_cost=training_cost,
                               # training_mse=training_mse,
                               testing_accuracy=testing_accuracy,
                               testing_cost=testing_cost,
                               testing_mse=testing_mse)
            server.update_time(time.time()-starting_time)

            self.training_accuracy_history.append(float(training_accuracy))
            # self.training_mse_history.append(training_mse)
            self.training_cost_history.append(training_cost)
            self.testing_accuracy_history.append(float(testing_accuracy))
            self.testing_mse_history.append(testing_mse)
            self.testing_cost_history.append(testing_cost)

            print('Epoch: ', epoch, "   Training Accuracy: ", training_accuracy, "   Test Accuracy: ", testing_accuracy,
                  "   Training cost: ", training_cost, "   Test Cost: ", testing_cost)

        self.accuracy_writer.update(epoch=self.plot_epoch,
                                    training_accuracy=self.training_accuracy_history,
                                    training_cost=self.training_cost_history,
                                    training_mse=self.training_mse_history,
                                    testing_accuracy=self.testing_accuracy_history,
                                    testing_cost=self.testing_cost_history,
                                    testing_mse=self.testing_mse_history)
        self.accuracy_writer.write()

        correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        test_y_prediction = sess.run(model, feed_dict={x: test_x})

        print("Calculating test-split accuracy...")
        print(sess.run(accuracy, feed_dict={x: test_x, labels: test_y}))
        mse = tf.reduce_mean(tf.square(test_y_prediction - test_y))
        print("Calculating test-split MSE (Mean Squared Error)...")
        mse_ = sess.run(mse)
        print(mse_)

        print("Calculating overall accuracy (entire dataset)...")
        print(sess.run(accuracy, feed_dict={x: sess.run(tf.reshape(X, [X.shape[0], self.height, self.width, 3])), labels: Y}))
        print("Calculating overall MSE (Mean Squared Error)...")
        print(sess.run(mse))

        self.store_model(sess)

    def store_model(self, sess):
        saver = tf.train.Saver()
        print("Storing model...")
        saver.save(sess, external_working_directory_path + 'trained_models/' + self.model_store_path + '/' + self.model_name)

    def display_progress(self):
        style.use('seaborn')
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_title('Accuracy', color='C0')

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        ax.plot(self.plot_epoch, self.training_accuracy_history, 'C1', label='Training accuracy')
        ax.plot(self.plot_epoch, self.testing_accuracy_history, 'C2', label='Testing accuracy')
        ax.legend()
        plt.show()

        style.use('seaborn')
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_title('Loss', color='C0')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        ax.plot(self.plot_epoch, self.training_cost_history, 'C1', label='Training cost')
        ax.plot(self.plot_epoch, self.testing_cost_history, 'C2', label='Testing cost')
        ax.legend()
        plt.show()

    def augment_data(self, data, labels):
        print("Started data augmentation...")
        augmented_data = tf.constant([], tf.float32, shape=[0, 10, 10, 3])
        augmented_labels = tf.constant([], tf.float32, shape=[0, 3])
        for data_anum in range(5):
            print(data_anum, "N")
            for n_image, n_label in zip(range(data.shape[0]), range(labels.shape[0])):
                flip_1 = tf.image.flip_up_down(data[n_image])
                flip_2 = tf.image.flip_left_right(data[n_image])
                flip_3 = tf.image.random_flip_up_down(data[n_image])
                flip_4 = tf.image.random_flip_left_right(data[n_image])

                pre_data = tf.concat([tf.expand_dims(data[n_image], 0), tf.expand_dims(flip_1, 0),
                                      tf.expand_dims(flip_2, 0), tf.expand_dims(flip_3, 0),
                                      tf.expand_dims(flip_4, 0)], 0)

                pre_labels = tf.concat([tf.expand_dims(labels[n_label], 0), tf.expand_dims(labels[n_label], 0),
                                        tf.expand_dims(labels[n_label], 0), tf.expand_dims(labels[n_label], 0),
                                        tf.expand_dims(labels[n_label], 0)], 0)
                augmented_data = tf.concat([augmented_data, tf.cast(pre_data, tf.float32)], 0)
                augmented_labels = tf.concat([augmented_labels, tf.cast(pre_labels, tf.float32)], 0)

        return augmented_data, augmented_labels

    def check_data_type(self):
        if self.trainable != 'True':
            raise DataNotTrainableError('Data is not trainable: {}'.format(self.training_data_path))

