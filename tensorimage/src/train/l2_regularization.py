import tensorflow as tf
from tensorimage.ops.models import *
from tensorimage.src.weights_initializer import init_weights


class L2RegularizationBuilder:
    def __init__(self, architecture: classmethod, l2_reg_beta):
        self.architecture = architecture
        self.l2_reg_beta = l2_reg_beta

    def start(self):
        if self.architecture.__doc__ == RosNetModel.__doc__:
            return (self.l2_reg_beta * tf.nn.l2_loss(init_weights('weights', 'conv1', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('biases', 'conv1', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('weights', 'conv2', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('biases', 'conv2', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('weights', 'fcl', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('biases', 'fcl', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('weights', 'out', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('biases', 'out', None)))
        elif self.architecture.__doc__ == AlexNetModel.__doc__:
            return (self.l2_reg_beta * tf.nn.l2_loss(init_weights('weights', 'conv1', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('biases', 'conv1', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('weights', 'conv2', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('biases', 'conv2', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('weights', 'conv3', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('biases', 'conv3', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('weights', 'conv4', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('biases', 'conv4', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('weights', 'conv5', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('biases', 'conv5', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('weights', 'fcl', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('biases', 'fcl', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('weights', 'fcl2', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('biases', 'fcl2', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('weights', 'out', None)) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('biases', 'out', None)))
