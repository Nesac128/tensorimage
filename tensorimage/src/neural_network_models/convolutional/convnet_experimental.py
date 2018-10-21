import tensorflow as tf


def conv2d(x, weights):
    return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')


def max_pool2d(x, kSizeX: int, kSizeY: int, strideX, strideY: int):
    return tf.nn.max_pool(x, ksize=[1, kSizeX, kSizeY, 1], strides=[1, strideX, strideY, 1], padding='SAME')


def convolutional_neural_network(data, weights, biases):
    dropout_prob = tf.constant(0.5, tf.float32)

    conv1 = conv2d(data, weights['conv_weights1'] + biases['conv_biases1'])
    print("Conv1 ", conv1.shape)
    max_pool1 = max_pool2d(conv1, kSizeX=2, kSizeY=2, strideX=2, strideY=2)
    print("Max Pool1 ", max_pool1.shape)

    conv2 = conv2d(max_pool1, weights['conv_weights2'] + biases['conv_biases2'])
    print("Conv2 ", conv2.shape)
    max_pool2 = max_pool2d(conv2, kSizeX=3, kSizeY=3, strideX=3, strideY=3)
    print("Max Pool2 ", max_pool2.shape)

    conv3 = conv2d(max_pool2, weights['conv_weights3'] + biases['conv_biases3'])
    print("Conv3 ", conv3.shape)
    max_pool3 = max_pool2d(conv3, kSizeX=3, kSizeY=3, strideX=2, strideY=2)
    print("Max Pool3 ", max_pool3.shape)

    conv4 = conv2d(max_pool3, weights['conv_weights4'] + biases['conv_biases4'])
    print("Conv4 ", conv4.shape)
    max_pool4 = max_pool2d(conv4, kSizeX=3, kSizeY=3, strideX=2, strideY=2)
    print("Max Pool4 ", max_pool4.shape)

    conv5 = conv2d(max_pool4, weights['conv_weights5'] + biases['conv_biases5'])
    print("Conv5 ", conv5.shape)
    max_pool5 = max_pool2d(conv5, kSizeX=2, kSizeY=2, strideX=2, strideY=2)
    print("Max Pool5 ", max_pool5.shape)

    dropout = tf.nn.dropout(max_pool5, dropout_prob)

    fcl6 = tf.reshape(dropout, [tf.shape(data)[0], int(list(dropout.shape)[1]) *
                                int(list(dropout.shape)[2]) *
                                int(list(dropout.shape)[3])])
    print(fcl6.shape, " FCL6")
    fcl6 = tf.nn.relu(tf.add(tf.matmul(fcl6, weights['fcl_weights6']), biases['fcl_biases6']))

    dropout2 = tf.nn.dropout(fcl6, dropout_prob)

    fcl7 = tf.nn.relu(tf.add(tf.matmul(dropout2, weights['fcl_weights7']), biases['fcl_biases7']))

    output = tf.add(tf.matmul(fcl7, weights['out_weights8']), biases['out_biases8'])

    return output
