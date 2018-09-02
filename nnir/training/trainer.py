import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib

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
                 train_test_split: float = 0.1,
                 training_type='conv'):
        self.labels = []

        # Store parameter inputs
        self.wsid = sess_id
        self.model_store_path = model_store_path
        self.model_fname = model_name
        self.optimizer = optimizer
        self.n_perceptrons_layer = n_perceptrons_layer
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_test_split = train_test_split
        self.training_type = training_type

        self.Meta = MetaData()
        raw_meta = self.Meta.read('data_path', 'n_columns', 'n_classes', 'trainable', 'type', 'data_len', sess_id=sess_id)

        meta = [mt for mt in raw_meta]

        self.n_columns = int(meta[0])
        self.training_data_path = meta[1]
        self.n_classes = int(meta[2])
        self.trainable = meta[3]
        self.type = meta[4]
        self.data_len = int(meta[5])
        if self.trainable == 'True':
            pass
        else:
            raise DataNotTrainableError('Data is not trainable: {}'.format(self.training_data_path))

    def read_dataset(self):
        df = pd.read_csv(self.training_data_path, header=None)
        X = df[df.columns[0:self.data_len]].values
        Y = df[df.columns[self.data_len:self.n_columns]]

        print(X.shape, "X shape")
        print(Y.shape, "Y shape")

        global labels
        labels = pd.Series.tolist(Y)

        encoder = LabelEncoder()
        encoder.fit(Y)
        y = encoder.transform(Y)
        Y = self.one_hot_encoder(y)

        return X, Y

    def one_hot_encoder(self, dlabels):
        n_labels = len(dlabels)
        n_unique_labels = len(np.unique(dlabels))
        one_hot_encode = np.zeros((n_labels, n_unique_labels))
        one_hot_encode[np.arange(n_labels), dlabels] = 1
        return one_hot_encode

    def train(self):
        global_step = tf.Variable(self.epochs, trainable=False, name='global_step')

        sess = tf.Session()

        X, Y = self.read_dataset()

        X, Y = shuffle(X, Y, random_state=1)
        # X = tf.divide(X,255)
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=self.train_test_split, random_state=415)

        if self.training_type == 'conv':
            train_x = sess.run(tf.reshape(train_x, shape=[train_x.shape[0], 10, 10, 3]))
            test_x = sess.run(tf.reshape(test_x, shape=[test_x.shape[0], 10, 10, 3]))

            # train_x, train_y = self.augment_data(train_x, train_y)
            # train_x, train_y = self.augment_data(train_x, train_y)
            # train_x, train_y = sess.run(self.augment_data(train_x, train_y))
            # print(train_x.shape, train_y.shape)

            # test_x, test_y = self.augment_data(test_x, test_y)
            # test_x, test_y = self.augment_data(test_x, test_y)
            # test_x, test_y = sess.run(self.augment_data(test_x, test_y))
            # print(test_x.shape, test_y.shape)

        n_dim = tf.constant(X.shape[1], name='n_dim')

        n_hidden_1 = self.n_perceptrons_layer[0]
        n_hidden_2 = self.n_perceptrons_layer[1]
        n_hidden_3 = self.n_perceptrons_layer[2]
        output_layer = self.n_perceptrons_layer[3]

        if self.training_type == 'fcl':
            weights = {
                'h1': tf.Variable(tf.truncated_normal([sess.run(n_dim), n_hidden_1]), name='weights1'),
                'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2]), name='weights2'),
                'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3]), name='weights3'),
                'h4': tf.Variable(tf.truncated_normal([n_hidden_3, output_layer]), name='weights4'),
                'out': tf.Variable(tf.truncated_normal([output_layer, self.n_classes]), name='weights5')
            }
            biases = {
                'b1': tf.Variable(tf.truncated_normal([n_hidden_1]), name='biases1'),
                'b2': tf.Variable(tf.truncated_normal([n_hidden_2]), name='biases2'),
                'b3': tf.Variable(tf.truncated_normal([n_hidden_3]), name='biases3'),
                'b4': tf.Variable(tf.truncated_normal([output_layer]), name='biases4'),
                'out': tf.Variable(tf.truncated_normal([self.n_classes]), name='biases5')
            }
        elif self.training_type == 'conv':
            weights = {'conv_weights1': tf.Variable(tf.random_normal([3, 3, 3, 3]), name='conv_weights1'),
                       'conv_weights2': tf.Variable(tf.random_normal([3, 3, 3, 1]), name='conv_weights2'),
                       # 'conv_weights3': tf.Variable(tf.random_normal([2, 2, 64, 32])),
                       # 'conv_weights4': tf.Variable(tf.random_normal([2, 2, 128, 256])),
                       # 'conv_weights5': tf.Variable(tf.random_normal([2, 2, 256, 8])),
                       'fcl_weights3': tf.Variable(tf.random_normal([100, 1024]), name='fcl_weights3'),
                       'out_weights4': tf.Variable(tf.random_normal([1024, self.n_classes]), name='out_weights4')
                       }
            biases = {'conv_biases1': tf.Variable(tf.random_normal([3]), name='conv_biases1'),
                      'conv_biases2': tf.Variable(tf.random_normal([1]), name='conv_biases2'),
                      # 'conv_biases3': tf.Variable(tf.random_normal([32])),
                      # 'conv_biases4': tf.Variable(tf.random_normal([256])),
                      # 'conv_biases5': tf.Variable(tf.random_normal([8])),
                      'fcl_biases3': tf.Variable(tf.random_normal([1024]), name='fcl_biases3'),
                      'out_biases4': tf.Variable(tf.random_normal([self.n_classes]), name='out_biases4')
                      }
        else:
            weights = None
            biases = None
            exit(1)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        model = None
        keep_prob = None
        x = None
        if self.training_type == 'fcl':
            x = tf.placeholder(tf.float32, [None, sess.run(n_dim)], name='x')
            print(x.shape)
            model = multilayer_perceptron(x, weights, biases)
        elif self.training_type == 'conv':
            x = tf.placeholder(tf.float32, [None, 10, 10, 3], name='x')
            print(x.shape)
            model = convolutional_neural_network(x, weights, biases)

        labels = tf.placeholder(tf.float32, [None, self.n_classes])

        # cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=labels))
        cost_function = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=model, labels=labels)) +
                0.01 * tf.nn.l2_loss(weights['conv_weights1']) +
                0.01 * tf.nn.l2_loss(biases['conv_biases1']) +
                0.01 * tf.nn.l2_loss(weights['conv_weights2']) +
                0.01 * tf.nn.l2_loss(biases['conv_biases2']) +
                0.01 * tf.nn.l2_loss(weights['fcl_weights3']) +
                0.01 * tf.nn.l2_loss(biases['fcl_biases3']) +
                0.01 * tf.nn.l2_loss(weights['out_weights4']) +
                0.01 * tf.nn.l2_loss(biases['out_biases4']))

        training_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost_function)

        plot_epoch = []
        mse_history = []
        training_loss_history = []
        testing_loss_history = []
        training_accuracy_history = []
        testing_accuracy_history = []

        sess.run(tf.global_variables_initializer())

        for epoch in range(self.epochs):
            if epoch != 0:
                if epoch % 50 == 0:
                    matplotlib.style.use('seaborn')
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.set_title('Accuracy', color='C0')

                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy')

                    ax.plot(plot_epoch, training_accuracy_history, 'C1', label='Training accuracy')
                    ax.plot(plot_epoch, testing_accuracy_history, 'C2', label='Testing accuracy')
                    ax.legend()
                    plt.show()

                    matplotlib.style.use('seaborn')
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.set_title('Loss', color='C0')

                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')

                    ax.plot(plot_epoch, training_loss_history, 'C1', label='Training cost')
                    ax.plot(plot_epoch, testing_loss_history, 'C2', label='Testing cost')
                    ax.legend()
                    plt.show()
                    # plt.plot(plot_epoch, training_accuracy_history, label='Training accuracy')
                    # plt.plot(plot_epoch, testing_accuracy_history, label='Testing accuracy')
                    # plt.show()
                    # plt.plot(plot_epoch, t)

            plot_epoch.append(epoch)

            sess.run(training_step, feed_dict={x: train_x, labels: train_y})
            correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(labels, 1))
            pred_y = sess.run(model, feed_dict={x: test_x})

            mse = tf.reduce_mean(tf.square(pred_y - test_y))
            mse_ = sess.run(mse)
            mse_history.append(mse_)

            training_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            training_accuracy = (sess.run(training_accuracy, feed_dict={x: train_x, labels: train_y}))
            training_accuracy_history.append(training_accuracy)

            training_cost = sess.run(cost_function, feed_dict={x: train_x, labels: train_y})
            training_loss_history.append(training_cost)

            testing_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            testing_accuracy = (sess.run(testing_accuracy, feed_dict={x: test_x, labels: test_y}))
            testing_accuracy_history.append(testing_accuracy)

            testing_cost = sess.run(cost_function, feed_dict={x: test_x, labels: test_y})
            testing_loss_history.append(testing_cost)

            # data = tf.cast(tf.reshape(train_x, shape=[tf.shape(train_x)[0], 10, 10, 3]), tf.float32)
            # conv1 = tf.nn.conv2d(data, weights['conv_weights1'], strides=[1,1,1,1], padding='SAME')
            # conv1 = tf.cast(tf.multiply(tf.nn.sigmoid(tf.abs(conv1)),255), tf.int32)
            # conv2 = tf.nn.conv2d(data, weights['conv_weights2'], strides=[1, 1, 1, 1], padding='SAME')
            # conv2 = tf.cast(tf.multiply(tf.nn.sigmoid(tf.abs(conv2)), 255), tf.int32)
            # # print(sess.run(conv1))
            # # exit()
            # timg = [(), (), (), (), (), (), (), (), (), ()]
            # for rwn in range(conv1.shape[1]):
            #     for cln in range(conv1.shape[2]):
            #         timg[rwn] += tuple(sess.run(conv1[0, rwn, cln]))
            # timg2 = [(), (), (), (), (), (), (), (), (), ()]
            # for rwn in range(conv2.shape[1]):
            #     for cln in range(conv2.shape[2]):
            #         timg2[rwn] += tuple(sess.run(conv2[0, rwn, cln]))
            # print(timg)
            # # exit()
            # file = open('/home/planetgazer8360/Desktop/Conv1FilterImages4/conv1_filter'+str(epoch)+'.png', 'wb')
            # writer = png.Writer(10, 10)
            # writer.write(file, timg)
            # file.close()
            #
            # file = open('/home/planetgazer8360/Desktop/Conv1FilterImages4/conv2_filter' + str(epoch) + '.png', 'wb')
            # writer = png.Writer(10, 10)
            # writer.write(file, timg2)
            # file.close()

            print('Epoch: ', epoch, "   Training Accuracy: ", training_accuracy, "   Test Accuracy: ", testing_accuracy,
                  "   Training cost: ", training_cost, "   Test Cost ", testing_cost)

        correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        test_y_prediction = sess.run(model, feed_dict={x: test_x})
        prediction_all = sess.run(model, feed_dict={x: sess.run(tf.reshape(X, [X.shape[0], 10, 10, 3]))})

        print("Calculating test-split accuracy...")
        print(sess.run(accuracy, feed_dict={x: test_x, labels: test_y}))
        mse = tf.reduce_mean(tf.square(test_y_prediction - test_y))
        print("Calculating test-split MSE (Mean Squared Error)...")
        mse_ = sess.run(mse)
        print(mse_)

        print("Calculating overall accuracy (entire dataset)...")
        print(sess.run(accuracy, feed_dict={x: sess.run(tf.reshape(X, [X.shape[0], 10, 10, 3])), labels: Y}))
        # mse = tf.reduce_mean(tf.square(tf.convert_to_tensor(prediction_all) - tf.convert_to_tensor(Y)))
        print("Calculating overall MSE (Mean Squared Error)...")
        print(sess.run(mse))

        print("Storing model...")
        saver.save(sess, external_working_directory_path+self.model_store_path+self.model_fname,
                   global_step=tf.train.global_step(sess, global_step))

        self.write_labels()
        self.Meta.write(self.wsid, trained_full_model_path=self.model_store_path+self.model_fname)

    def augment_data(self, data, labels):
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

    def write_labels(self):
        with open(external_working_directory_path+self.model_store_path+'training_labels.txt', 'w') as lbfile:
            for label in labels:
                lbfile.write(label[0] + '\n')
