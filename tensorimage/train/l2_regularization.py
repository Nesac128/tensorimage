import tensorflow as tf
from tensorimage.train.weights_initializer import init_weights


class L2RegularizationBuilder:
    def __init__(self, architecture: str, l2_reg_beta: float, shapes: tuple):
        self.architecture = architecture
        self.l2_reg_beta = l2_reg_beta
        self.ws = shapes[0]
        self.bs = shapes[1]

    def start(self):
        if self.architecture == 'RosNet':
            return (self.l2_reg_beta * tf.nn.l2_loss(init_weights('weights', 'conv1', self.ws["conv1"])) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('biases', 'conv1', self.bs["conv1"])) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('weights', 'conv2', self.ws["conv2"])) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('biases', 'conv2', self.bs["conv2"])) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('weights', 'fcl', self.ws["fcl"])) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('biases', 'fcl', self.bs["fcl"])) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('weights', 'out', self.ws["out"])) +
                    self.l2_reg_beta * tf.nn.l2_loss(init_weights('biases', 'out', self.bs["out"])))
        elif self.architecture == 'AlexNet':
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
