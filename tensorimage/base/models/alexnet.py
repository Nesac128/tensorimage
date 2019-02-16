from tensorimage.base.ops import *
from tensorimage.base.weights_initializer import init_weights
from tensorimage.base.display_architecture import display_architecture


class AlexNet:
    """alexnet"""
    def __init__(self, height, width, n_channels, n_classes):
        self.height = height
        self.width = width
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.layer_names = ["conv1", "conv2", "conv3", "conv4", "conv5", "fcl", "fcl2", "out"]
        self.weights_shapes = {
            self.layer_names[0]: [11, 11, 3, 96],
            self.layer_names[1]: [5, 5, 96, 256],
            self.layer_names[2]: [3, 3, 256, 384],
            self.layer_names[3]: [3, 3, 384, 384],
            self.layer_names[4]: [3, 3, 384, 256],
            self.layer_names[5]: [8 * 8 * 256, 4096],
            self.layer_names[6]: [4096, 4096],
            self.layer_names[7]: [4096, self.n_classes]
        }
        self.biases_shapes = {
            self.layer_names[0]: [96],
            self.layer_names[1]: [256],
            self.layer_names[2]: [384],
            self.layer_names[3]: [384],
            self.layer_names[4]: [256],
            self.layer_names[5]: [4096],
            self.layer_names[6]: [4096],
            self.layer_names[7]: [self.n_classes]
        }

    def convnet(self, x):
        x = tf.image.resize_images(x, tf.constant([227, 227]))
        with tf.name_scope('alexnet'):
            with tf.name_scope('conv1_layer'):
                with tf.name_scope('conv1'):
                    conv1 = conv2d(x, init_weights('weights', 'conv1', [11, 11, 3, 96]) +
                                   init_weights('biases', 'conv1', [96]), strides=(4, 4))
                    print(conv1.shape, " Conv1")
                with tf.name_scope('conv1_maxpool2d'):
                    conv1_maxpool2d = maxpool2d(conv1, strides=(2, 2), filter_sizes=(3, 3))
                    print(conv1_maxpool2d.shape, " Conv1 MaxPool")
            with tf.name_scope('conv2_layer'):
                with tf.name_scope('conv2'):
                    conv2 = conv2d(conv1_maxpool2d, init_weights('weights', 'conv2', [5, 5, 96, 256]) +
                                   init_weights('biases', 'conv2', [256]), strides=(1, 1))
                    print(conv2.shape, " Conv2")
                with tf.name_scope('conv2_maxpool2d'):
                    conv2_maxpool2d = maxpool2d(conv2, strides=(2, 2), filter_sizes=(3, 3))
                    print(conv2_maxpool2d.shape, " Conv2 MaxPool")
            with tf.name_scope('conv3_layer'):
                with tf.name_scope('conv3'):
                    conv3 = conv2d(conv2_maxpool2d, init_weights('weights', 'conv3', [3, 3, 256, 384]) +
                                   init_weights('biases', 'conv3', [384]), strides=(1, 1))
                    print(conv3.shape, " Conv3")
            with tf.name_scope('conv4_layer'):
                with tf.name_scope('conv4'):
                    conv4 = conv2d(conv3, init_weights('weights', 'conv4', [3, 3, 384, 384]) +
                                   init_weights('biases', 'conv4', [384]), strides=(1, 1))
                    print(conv4.shape, " Conv4")
            with tf.name_scope('conv5_layer'):
                with tf.name_scope('conv5'):
                    conv5 = conv2d(conv4, init_weights('weights', 'conv5', [3, 3, 384, 256]) +
                                   init_weights('biases', 'conv5', [256]), strides=(1, 1))
                    print(conv5.shape, " Conv5")
                with tf.name_scope('conv5_maxpool2d'):
                    conv5_maxpool2d = maxpool2d(conv5, strides=(2, 2), filter_sizes=(3, 3))
                    print(conv5_maxpool2d.shape, " Conv5 MaxPool")
            with tf.name_scope('fcl_layer'):
                with tf.name_scope('flatten'):
                    fclr = tf.reshape(conv5_maxpool2d, [tf.shape(x)[0], 8*8*256])
                    print(fclr.shape, " FCL reshape")
                with tf.name_scope('ReLU_add_matmul'):
                    fcl = tf.nn.relu(tf.add(tf.matmul(fclr, init_weights('weights', 'fcl', [8 * 8 * 256, 4096])),
                                            init_weights('biases', 'fcl', [4096])))
                    print(fcl.shape, " FCL")
            with tf.name_scope('fcl2_layer'):
                with tf.name_scope('ReLU_add_matmul'):
                    fcl2 = tf.nn.relu(tf.add(tf.matmul(fcl, init_weights('weights', 'fcl2', [4096, 4096])),
                                             init_weights('biases', 'fcl2', [4096])))
                    print(fcl2.shape, " FCL 2")
            with tf.name_scope('out_layer'):
                model = tf.add(tf.matmul(fcl2, init_weights('weights', 'out', [4096, self.n_classes])),
                               init_weights('biases', 'out', [self.n_classes]))
                print(model.shape, " OUTPUT")

        display_architecture(Conv1=conv1.shape,
                             Conv1MaxPool2d=conv1_maxpool2d.shape,
                             Conv2=conv2.shape,
                             Conv2MaxPool2d=conv2_maxpool2d.shape,
                             Conv3=conv3.shape,
                             Conv4=conv4.shape,
                             Conv5=conv5.shape,
                             Conv5MaxPool2d=conv5_maxpool2d.shape,
                             FCL_flatten=fclr.shape,
                             FCL=fcl.shape,
                             FCL2=fcl2.shape,
                             OutputLayer=model.shape)
        return model
