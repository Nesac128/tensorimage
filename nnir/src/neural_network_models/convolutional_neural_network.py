import tensorflow as tf


def conv2d(x, weights):
    return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')


def max_pool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')


def convolutional_neural_network(data, weights, biases, n_nodes):
    conv1 = conv2d(data, weights['conv_weights1'] + biases['conv_biases1'])

    # conv1 = max_pool2d(conv1)
    conv2 = conv2d(conv1, weights['conv_weights2'] + biases['conv_biases2'])
    # conv2 = max_pool2d(conv2)

    # conv3 = conv2d(conv2, weights['conv_weights3']+biases['conv_biases3'])

    # conv4 = conv2d(conv3, weights['conv_weights4']+biases['conv_biases4'])
    #
    # conv5 = conv2d(conv4, weights['conv_weights5']+biases['conv_biases5'])

    fcl = tf.reshape(conv2, [tf.shape(data)[0], n_nodes])
    fcl = tf.nn.relu(tf.add(tf.matmul(fcl, weights['fcl_weights3']), biases['fcl_biases3']))

    output = tf.add(tf.matmul(fcl, weights['out_weights4']), biases['out_biases4'])

    return output
