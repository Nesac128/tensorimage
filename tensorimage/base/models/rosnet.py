from tensorimage.base.ops import *
from tensorimage.base.weights_initializer import init_weights
from tensorimage.base.display_architecture import display_architecture


class RosNet:
    """rosnet"""
    def __init__(self, x, n_classes):
        self.x = x
        self.n_classes = n_classes

        height, width = self.x.shape[1], self.x.shape[2]
        self.shape = [1, height, width, 3]
        self.rv = self.get_rv(self.shape)

        self.layer_names = ["conv1", "conv2", "fcl", "out"]

        self.weights_shapes = {
            self.layer_names[0]: [5, 5, 3, 3],
            self.layer_names[1]: [3, 3, 3, 64],
            self.layer_names[2]: [self.rv*64, 128],
            self.layer_names[3]: [128, self.n_classes]
        }
        self.biases_shapes = {
            "conv1": [3],
            "conv2": [64],
            "fcl": [128],
            "out": [self.n_classes]
        }

    def convnet(self):
        with tf.name_scope('rosnet'):
            with tf.name_scope('conv1_layer'):
                with tf.name_scope('conv1'):
                    conv1 = conv2d(self.x, init_weights('weights', 'conv1', [5, 5, 3, 3]) +
                                   init_weights('biases', 'conv1', [3]))
                with tf.name_scope('conv1_maxpool2d'):
                    conv1_maxpool2d = maxpool2d(conv1)
            with tf.name_scope('conv2_layer'):
                with tf.name_scope('conv2'):
                    conv2 = conv2d(conv1_maxpool2d, init_weights('weights', 'conv2', [3, 3, 3, 64]) +
                                   init_weights('biases', 'conv2', [64]))
                with tf.name_scope('conv2_maxpool2d'):
                    conv2_maxpool2d = maxpool2d(conv2)
            with tf.name_scope('fcl_layer'):
                with tf.name_scope('flatten'):
                    fclr = tf.reshape(conv2_maxpool2d, [tf.shape(self.x)[0], self.rv*64])
                with tf.name_scope('ReLU_add_matmul'):
                    fcl = tf.nn.relu(tf.add(tf.matmul(fclr, init_weights('weights', 'fcl', [self.rv*64, 128])),
                                            init_weights('biases', 'fcl', [128])))
            with tf.name_scope('out_layer'):
                model = tf.add(tf.matmul(fcl, init_weights('weights', 'out', [128, self.n_classes])),
                               init_weights('biases', 'out', [self.n_classes]))

        display_architecture(Conv1=conv1.shape,
                             Conv1_MaxPool2d=conv1_maxpool2d.shape,
                             Conv2=conv2.shape,
                             Conv2_MaxPool2d=conv2_maxpool2d.shape,
                             FCL_flatten=fclr.shape,
                             FCL=fcl.shape,
                             OutputLayer=model.shape)
        return model

    @staticmethod
    def get_rv(shape):
        conv1 = conv2d(tf.ones(shape), init_weights('rv', 'conv1', [5, 5, 3, 3]))
        conv1_maxpool2d = maxpool2d(conv1)
        conv2 = conv2d(conv1_maxpool2d, init_weights('rv', 'conv2', [3, 3, 3, 64]))
        conv2_maxpool2d = maxpool2d(conv2)
        return conv2_maxpool2d.shape[1]*conv2_maxpool2d.shape[2]
