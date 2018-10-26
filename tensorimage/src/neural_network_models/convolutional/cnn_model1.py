import tensorflow as tf


def conv2d(x, weights):
    return tf.nn.conv2d(x, weights, strides=[1, 2, 2, 1], padding='SAME')


def max_pool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(data, weights, biases):
    with tf.name_scope('conv1_layer'):
        with tf.name_scope('conv1'):
            conv1 = conv2d(data, weights['conv1'] + biases['conv1'])
            print(conv1.shape, " Conv1")
        with tf.name_scope('conv1_maxpool2d'):
            conv1_maxpool2d = max_pool2d(conv1)
            print(conv1_maxpool2d.shape, " Conv1 MaxPool")

    with tf.name_scope('conv2_layer'):
        with tf.name_scope('conv2'):
            conv2 = conv2d(conv1_maxpool2d, weights['conv2'] + biases['conv2'])
            print(conv2.shape, " Conv2")
        with tf.name_scope('conv2_maxpool2d'):
            conv2_maxpool2d = max_pool2d(conv2)
            print(conv2_maxpool2d.shape, " Conv2 MaxPool")

    with tf.name_scope('fcl_layer'):
        with tf.name_scope('flatten'):
            fcl = tf.reshape(conv2_maxpool2d, [tf.shape(data)[0], tf.shape(conv2_maxpool2d)[1] *
                                               tf.shape(conv2_maxpool2d)[2]*64])
            print(fcl.shape, " FCL reshape")
        with tf.name_scope('ReLU_add_matmul'):
            fcl = tf.nn.relu(tf.add(tf.matmul(fcl, weights['fcl']), biases['fcl']))
            print(fcl.shape, " FCL")

    with tf.name_scope('out_layer'):
        model = tf.add(tf.matmul(fcl, weights['out']), biases['out'])
        print(model.shape, " OUTPUT")
    return model
