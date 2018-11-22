import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from tensorimage.config import *
from tensorimage.src.file.reader import *
from tensorimage.src.file.writer import *
from tensorimage.src.train.convnet_builder import ConvNetBuilder
from tensorimage.src.train.l2_regularization import L2RegularizationBuilder


class Train:
    def __init__(self,
                 data_name: str,
                 training_name: str,
                 n_epochs: int,
                 learning_rate: float,
                 l2_regularization_beta: float,
                 architecture: str,
                 data_augmentation_builder: tuple=(None, False),
                 batch_size: int = 32,
                 train_test_split: float = 0.2):
        """
        :param data_name: unique name which identifies which image data to read for training
        :param n_epochs: number of epochs
        :param learning_rate: learning rate for optimizer
        :param l2_regularization_beta: value to use for L2 Regularization beta
        :param architecture: one of the CNN class architectures located in models/ directory
        :param data_augmentation_builder: tuple containing data augmentation builder class and
        boolean which specifies if to augment or not testing data
        :param train_test_split: proportion of data that will be used as testing set
        """
        self.data_name = data_name
        self.training_name = training_name
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.train_test_split = train_test_split
        self.l2_beta = l2_regularization_beta
        self.architecture = architecture
        self.data_augmentation_builder = data_augmentation_builder
        self.batch_size = batch_size

        self.training_metadata_writer = JSONWriter(self.training_name, training_metafile_path)

        self.metadata_reader = JSONReader(self.data_name, dataset_metafile_path)
        self.metadata_reader.bulk_read()
        self.metadata_reader.select()
        image_metadata = self.metadata_reader.selected_data
        self.n_columns = image_metadata["n_columns"]
        self.training_data_path = image_metadata["data_path"]
        self.n_classes = image_metadata["n_classes"]
        self.dataset_type = image_metadata["dataset_type"]
        assert self.dataset_type == 'training'
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
        self.test_x = None
        self.test_y = None

        self.augmented_train_x = None
        self.augmented_train_y = None
        self.augmented_test_x = None
        self.augmented_test_y = None

        self.convnet_builder = ConvNetBuilder(self.architecture)
        self.convolutional_neural_network = self.convnet_builder.build_convnet()
        self.l2_regularization_builder = L2RegularizationBuilder(self.architecture, self.l2_beta)
        self.l2_regularization = self.l2_regularization_builder.start()

        self.sess = tf.Session()

    def build_dataset(self):
        self.csv_reader.read_training_dataset(self.data_len, self.n_columns)
        self.X = self.csv_reader.X
        Y = self.csv_reader.Y

        print(self.X.shape, " X shape")
        print(Y.shape, " Y shape")

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

        if not self.data_augmentation_builder[0]:
            self.augmented_train_x, self.augmented_test_y = self.data_augmentation_builder[0].start(self.train_x,
                                                                                                    self.train_y,
                                                                                                    self.n_classes,
                                                                                                    (self.width,
                                                                                                     self.height),
                                                                                                    n_channels)

    def train(self):
        x = tf.placeholder(tf.float32, [None, self.height, self.width, 3], name='x')
        labels = tf.placeholder(tf.float32, [None, self.n_classes])

        batch_iters = self.train_x.shape[0] // self.batch_size

        init = tf.global_variables_initializer()
        self.sess.run(init)

        convnet = self.convolutional_neural_network(x, self.n_classes)
        model = convnet.convnet()

        with tf.name_scope('cost_function'):
            cost_function = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=model, labels=labels)) + self.l2_regularization)

        training_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cost_function)

        self.sess.run(tf.global_variables_initializer())

        correct_prediction_ = tf.equal(tf.argmax(model, 1), tf.argmax(labels, 1))
        training_accuracy = tf.reduce_mean(tf.cast(correct_prediction_, tf.float32))
        testing_accuracy = tf.reduce_mean(tf.cast(correct_prediction_, tf.float32))

        with tf.name_scope('accuracy'):
            tf.summary.scalar('training_accuracy', training_accuracy)
            tf.summary.scalar('testing_accuracy', testing_accuracy)
        with tf.name_scope('cost'):
            tf.summary.scalar('training_cost', cost_function)

        write_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(workspace_dir+'user/logs/'+str(self.training_name), self.sess.graph)
        for epoch in range(1, self.n_epochs+1):
            training_accuracy_ = 0
            testing_accuracy_ = 0
            training_cost = 0
            testing_cost = 0
            try:
                for i in range(1, batch_iters+1):
                    x_, y = self._get_batch(i)
                    self.sess.run(training_step, feed_dict={x: x_, labels: y})

                    batch_training_accuracy = (self.sess.run(training_accuracy, feed_dict={x: x_, labels: y}))
                    batch_testing_accuracy = (self.sess.run(testing_accuracy, feed_dict={x: self.test_x, labels: self.test_y}))
                    batch_testing_cost = self.sess.run(cost_function, feed_dict={x: self.test_x, labels: self.test_y})
                    batch_training_cost = self.sess.run(cost_function, feed_dict={x: x_, labels: y})

                    training_accuracy_ += batch_training_accuracy
                    testing_accuracy_ += batch_testing_accuracy
                    training_cost += batch_training_cost
                    testing_cost += batch_testing_cost
            except KeyboardInterrupt:
                print("\nYou have chosen to early stop the training process.")
                if self._confirm_early_stop():
                    break
            avr_training_accuracy = training_accuracy_/batch_iters
            avr_testing_accuracy = testing_accuracy_/batch_iters

            summary = self.sess.run(write_op, {training_accuracy: avr_training_accuracy,
                                               testing_accuracy: avr_testing_accuracy,
                                               cost_function: training_cost})
            writer.add_summary(summary, epoch)
            writer.flush()
            if epoch % 50 == 0:
                print("Epoch: ", epoch, "   Training accuracy = ", float("%0.5f" % avr_training_accuracy),
                      "   Testing accuracy = ", float("%0.5f" % avr_testing_accuracy), "   Training cost = ",
                      training_cost, "   Testing cost = ", testing_cost)

        self._store_model()
        self._write_metadata()
        writer.close()

    def _get_batch(self, i):
        return self.train_x[(self.batch_size*i)-self.batch_size:self.batch_size*i], \
                   self.train_y[(self.batch_size*i)-self.batch_size:self.batch_size*i]

    def _write_metadata(self):
        self.training_metadata_writer.update(data_name=self.data_name,
                                             model_folder_name=self.model_folder_name,
                                             model_name=self.training_name,
                                             dataset_name=self.dataset_name,
                                             architecture=self.architecture)
        self.training_metadata_writer.write()

    def _store_model(self):
        saver = tf.train.Saver()
        saver.save(self.sess, base_trained_models_store_path + self.model_folder_name
                   + '/' + self.training_name)

    def _confirm_early_stop(self):
        confirmation = input("Confirm early stopping [Y/n]: ")
        y = ['Y', 'y', 'yes', 'Yes']
        n = ['N', 'n', 'no', 'No']
        if confirmation in y:
            return True
        elif confirmation in n:
            return False
        else:
            print("Invalid option!")
            self._confirm_early_stop()

    @staticmethod
    def _one_hot_encode(dlabels):
        n_labels = len(dlabels)
        n_unique_labels = len(np.unique(dlabels))
        one_hot_encode = np.zeros((n_labels, n_unique_labels))
        one_hot_encode[np.arange(n_labels), dlabels] = 1
        return one_hot_encode
