import tensorflow as tf


def conv2d(x, weights):
    return tf.nn.conv2d(x, weights, strides=[1, 2, 2, 1], padding='SAME')


def max_pool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(data, weights, biases):
    dropout_prob = tf.constant(0.4, tf.float32)

    conv1 = conv2d(data, weights['conv_weights1'] + biases['conv_biases1'])
    print(conv1.shape, " Conv1")
    conv1 = max_pool2d(conv1)
    print(conv1.shape, " Conv1 MaxPool")

    # dropout = tf.nn.dropout(conv1, dropout_prob)

    conv2 = conv2d(conv1, weights['conv_weights2'] + biases['conv_biases2'])
    print(conv2.shape, " Conv2")
    conv2 = max_pool2d(conv2)
    print(conv2.shape, " Conv2 MaxPool")

    # dropout = tf.nn.dropout(conv2, dropout_prob)

    fcl = tf.reshape(conv2, [tf.shape(data)[0], 2*2*64])
    print(fcl.shape, " FCL reshape")
    fcl = tf.nn.relu(tf.add(tf.matmul(fcl, weights['fcl_weights3']), biases['fcl_biases3']))
    print(fcl.shape, " FCL")

    output = tf.add(tf.matmul(fcl, weights['out_weights4']), biases['out_biases4'])
    print(output.shape, " OUTPUT")

    return output
