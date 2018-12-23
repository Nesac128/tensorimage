import numpy as np
import tensorflow as tf
from tensorimage.config.info import workspace_dir
from tensorimage.train.weights_initializer import init_weights


class ModelRestorer:
    def __init__(self, model_folder_name: str, model_name: str, architecture: str, sess):
        self.model_folder_name = model_folder_name
        self.model_name = model_name
        self.architecture = architecture
        self.sess = sess

    def start(self):
        if self.architecture == 'RosNet':
            self.restore_rosnet_model()
        elif self.architecture == 'AlexNet':
            self.restore_alexnet_model()

    def restore_rosnet_model(self):
        saver = tf.train.import_meta_graph(workspace_dir + 'user/trained_models/' + self.model_folder_name + '/' + self.model_name + '.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint(workspace_dir + 'user/trained_models/' + self.model_folder_name + '/./'))

        layers = ['conv1', 'conv2', 'fcl', 'out']
        with tf.variable_scope('RosNet', reuse=tf.AUTO_REUSE):
            with tf.name_scope('weights_restore'):
                for layer in layers:
                    ly = self.sess.run('weights/'+layer+':0')
                    ly_list = np.ndarray.tolist(ly)
                    init_weights('weights', layer, ly.shape, initializer=tf.initializers.constant(ly_list))
            with tf.name_scope('biases_restore'):
                for layer in layers:
                    ly = self.sess.run('biases/'+layer+':0')
                    ly_list = np.ndarray.tolist(ly)
                    init_weights('biases', layer, ly.shape, initializer=tf.initializers.constant(ly_list))

    def restore_alexnet_model(self):
        saver = tf.train.import_meta_graph(workspace_dir + 'user/trained_models/' + self.model_folder_name + '/' + self.model_name + '.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint(workspace_dir + 'user/trained_models/' + self.model_folder_name + '/./'))

        layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fcl', 'fcl2', 'out']
        with tf.name_scope('weights_restore'):
            for layer in layers:
                ly = self.sess.run('weights/'+layer+':0')
                ly_list = np.ndarray.tolist(ly)
                init_weights('weights', layer, ly.shape, initializer=tf.initializers.constant(ly_list))
        with tf.name_scope('biases_restore'):
            for layer in layers:
                ly = self.sess.run('biases/'+layer+':0')
                ly_list = np.ndarray.tolist(ly)
                init_weights('biases', layer, ly.shape, initializer=tf.initializers.constant(ly_list))
