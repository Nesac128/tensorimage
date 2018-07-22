import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from nnir.pcontrol import *
from config import external_working_directory_path
from exceptions import *

from scripts.multilayer_perceptron import multilayer_perceptron
from scripts.convolutional_neural_network import convolutional_neural_network


class Train:
    def __init__(self,
                 sess_id,
                 model_store_path,
                 model_name,
                 optimizer='GradientDescent',
                 n_perceptrons_layer: tuple = (100, 100, 100, 100),
                 epochs: int = 150,
                 learning_rate: float = 0.2,
                 train_test_split: float = 0.1):
        self.labels = []

        # Store parameter inputs
        self.model_store_path = model_store_path
        self.model_fname = model_name
        self.optimizer = optimizer
        self.n_perceptrons_layer = n_perceptrons_layer
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_test_split = train_test_split

        self.Meta = MetaData(sess_id)
        raw_meta = self.Meta.read('data_path', 'n_columns', 'n_classes', 'trainable', sess_id=sess_id)

        meta = [mt for mt in raw_meta]

        self.n_columns = int(meta[0])
        self.training_data_path = meta[1]
        self.n_classes = int(meta[2])
        self.trainable = meta[3]
        if self.trainable == 'True':
            pass
        else:
            raise DataNotTrainableError('Data is not trainable: {}'.format(self.training_data_path))

    def read_dataset(self):
        df = pd.read_csv(self.training_data_path, header=None)
        X = df[df.columns[0:self.n_columns-1]].values
        y = df[df.columns[-1]]
        global labels
        labels = pd.Series.tolist(y)

        # Encode the dependen   t variable
        encoder = LabelEncoder()
        encoder.fit(y)
        y = encoder.transform(y)
        Y = self.one_hot_encode(y)
        print(X.shape, " Training data shape")
        print(Y.shape, " Labels shape")
        print(Y)
        return X, Y

    def one_hot_encode(self, dlabels):
        n_labels = len(dlabels)
        n_unique_labels = len(np.unique(dlabels))
        one_hot_encode = np.zeros((n_labels, n_unique_labels))
        one_hot_encode[np.arange(n_labels), dlabels] = 1
        return one_hot_encode

    def train(self):
        global_step = tf.Variable(self.epochs, trainable=False, name='global_step')

        sess = tf.Session()

        X, Y = self.read_dataset()
        # sess.run(X)

        X, Y = shuffle(X, Y, random_state=1)

        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.075, random_state=415)

        cost_history = np.empty(shape=[1], dtype=float)
        n_dim = tf.constant(X.shape[1], name='n_dim')

        n_hidden_1 = self.n_perceptrons_layer[0]
        n_hidden_2 = self.n_perceptrons_layer[1]
        n_hidden_3 = self.n_perceptrons_layer[2]
        n_hidden_4 = self.n_perceptrons_layer[3]
        output_layer = self.n_perceptrons_layer[3]

        x = tf.placeholder(tf.float32, [None, sess.run(n_dim)], name='x')
        labels = tf.placeholder(tf.float32, [None, self.n_classes])
        print(x.shape)

        weights = {
            'h1': tf.Variable(tf.truncated_normal([sess.run(n_dim), n_hidden_1]), name='weights1'),
            'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2]), name='weights2'),
            'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3]), name='weights3'),
            'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4]), name='weights4'),
            'out': tf.Variable(tf.truncated_normal([output_layer, self.n_classes]), name='weights5')
        }

        biases = {
            'b1': tf.Variable(tf.truncated_normal([n_hidden_1]), name='biases1'),
            'b2': tf.Variable(tf.truncated_normal([n_hidden_2]), name='biases2'),
            'b3': tf.Variable(tf.truncated_normal([n_hidden_3]), name='biases3'),
            'b4': tf.Variable(tf.truncated_normal([n_hidden_4]), name='biases4'),
            'out': tf.Variable(tf.truncated_normal([self.n_classes]), name='biases5')
        }

        saver = tf.train.Saver()
        model = convolutional_neural_network(x, self.n_classes)

        cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=labels))
        training_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost_function)

        mse_history = []
        accuracy_history = []

        init = tf.global_variables_initializer()
        sess.run(init)

        # print(sess.run(weights['h1']))
        # print(sess.run(tf.cast(tf.convert_to_tensor(X),tf.float32)))
        # print(sess.run(tf.matmul(tf.cast(tf.convert_to_tensor(X),tf.float32),weights['h1'])))
        # print(sess.run(biases['b1']))
        # print(sess.run(biases['b1']).shape)
        # print(sess.run(tf.matmul(tf.cast(tf.convert_to_tensor(X),tf.float32),weights['h1'])).shape)
        # exit()

        for epoch in range(self.epochs):
            sess.run(training_step, feed_dict={x: train_x, labels: train_y})
            cost = sess.run(cost_function, feed_dict={x: train_x, labels: train_y})
            cost_history = np.append(cost_history, cost)
            correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            pred_y = sess.run(model, feed_dict={x: test_x})
            mse = tf.reduce_mean(tf.square(pred_y - test_y))
            mse_ = sess.run(mse)
            mse_history.append(mse_)
            accuracy = (sess.run(accuracy, feed_dict={x: train_x, labels: train_y}))
            accuracy_history.append(accuracy)

            print('EPOCH ', epoch, ' --COST ', cost, " --MSE ", mse_, " --TRAINING ACCURACY ", accuracy)

        correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        test_y_prediction = sess.run(model, feed_dict={x: test_x})
        prediction_all = sess.run(model, feed_dict={x: X})

        print("Calculating test-split accuracy...")
        print(sess.run(accuracy, feed_dict={x: test_x, labels: test_y}))
        mse = tf.reduce_mean(tf.square(test_y_prediction - test_y))
        print("Calculating test-split MSE (Mean Squared Error)...")
        mse_ = sess.run(mse)
        print(mse_)

        print("Calculating overall accuracy (entire dataset)...")
        print(sess.run(accuracy, feed_dict={x: X, labels: Y}))
        mse = tf.reduce_mean(tf.square(tf.convert_to_tensor(prediction_all) - tf.convert_to_tensor(Y)))
        print("Calculating overall MSE (Mean Squared Error)...")
        print(sess.run(mse))

        saver.save(sess, external_working_directory_path+self.model_store_path+self.model_fname,
                   global_step=tf.train.global_step(sess, global_step))

        self.write_labels()
        self.Meta.write(trained_full_model_path=self.model_store_path+self.model_fname)

    def write_labels(self):
        with open(external_working_directory_path+self.model_store_path+'training_labels.txt', 'w') as lbfile:
            for label in labels:
                lbfile.write(label + '\n')
