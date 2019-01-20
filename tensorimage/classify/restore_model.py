import numpy as np
import tensorflow as tf
from tensorimage.config.info import workspace_dir
from tensorimage.base.weights_initializer import init_weights
from tensorimage.base.models.map.model import model_map


class ModelRestorer:
    def __init__(self, model_folder_name: str, model_name: str, architecture: str, sess):
        self.model_folder_name = model_folder_name
        self.model_name = model_name
        self.architecture = architecture
        self.sess = sess

        self.layer_names = model_map[self.architecture].layer_names

    def start(self):
        saver = tf.train.import_meta_graph(workspace_dir + 'user/trained_models/' + self.model_folder_name + '/' + self.model_name + '.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint(workspace_dir + 'user/trained_models/' + self.model_folder_name + '/./'))

        with tf.variable_scope(self.architecture, reuse=tf.AUTO_REUSE):
            with tf.name_scope('weights_restore'):
                for layer in self.layer_names:
                    layer_weights = self.sess.run('weights/' + layer + ':0')
                    layer_weights_ = np.ndarray.tolist(layer_weights)
                    init_weights('weights', layer, layer_weights.shape, initializer=tf.initializers.constant(layer_weights_))
            with tf.name_scope('biases_restore'):
                for layer in self.layer_names:
                    layer_biases = self.sess.run('biases/' + layer + ':0')
                    layer_biases_ = np.ndarray.tolist(layer_biases)
                    init_weights('biases', layer, layer_biases.shape, initializer=tf.initializers.constant(layer_biases_))
