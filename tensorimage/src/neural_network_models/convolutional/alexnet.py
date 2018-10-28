import tensorflow as tf


def conv2d(x, weights, kStrideX, kStrideY):
    return tf.nn.conv2d(x, weights, strides=[1, kStrideX, kStrideY, 1], padding='SAME')


def max_pool2d(x, kStrideX, kStrideY, kSizeX, kSizeY):
    return tf.nn.max_pool(x, ksize=[1, kSizeX, kSizeY, 1], strides=[1, kStrideX, kStrideY, 1], padding='SAME')


def convolutional_neural_network(data, weights, biases):
    print("AlexNet Architecture")

    data = tf.image.resize_images(data, tf.constant([227, 227]))

    with tf.name_scope('conv1_layer'):
        with tf.name_scope('conv1'):
            conv1 = conv2d(data, weights['conv1'] + biases['conv1'], kStrideX=4, kStrideY=4)
            print(conv1.shape, " Conv1")
        with tf.name_scope('conv1_maxpool2d'):
            conv1_maxpool2d = max_pool2d(conv1, kStrideX=2, kStrideY=2, kSizeX=3, kSizeY=3)
            print(conv1_maxpool2d.shape, " Conv1 MaxPool")

    with tf.name_scope('conv2_layer'):
        with tf.name_scope('conv2'):
            conv2 = conv2d(conv1_maxpool2d, weights['conv2'] + biases['conv2'], kStrideX=1, kStrideY=1)
            print(conv2.shape, " Conv2")
        with tf.name_scope('conv2_maxpool2d'):
            conv2_maxpool2d = max_pool2d(conv2, kStrideX=2, kStrideY=2, kSizeX=3, kSizeY=3)
            print(conv2_maxpool2d.shape, " Conv2 MaxPool")

    with tf.name_scope('conv3_layer'):
        with tf.name_scope('conv3'):
            conv3 = conv2d(conv2_maxpool2d, weights['conv3'] + biases['conv3'], kStrideX=1, kStrideY=1)
            print(conv3.shape, " Conv3")

    with tf.name_scope('conv4_layer'):
        with tf.name_scope('conv4'):
            conv4 = conv2d(conv3, weights['conv4'] + biases['conv4'], kStrideX=1, kStrideY=1)
            print(conv4.shape, " Conv4")

    with tf.name_scope('conv5_layer'):
        with tf.name_scope('conv5'):
            conv5 = conv2d(conv4, weights['conv5'] + biases['conv5'], kStrideX=1, kStrideY=1)
            print(conv5.shape, " Conv5")
        with tf.name_scope('conv5_maxpool2d'):
            conv5_maxpool2d = max_pool2d(conv5, kStrideX=2, kStrideY=2, kSizeX=3, kSizeY=3)
            print(conv5_maxpool2d.shape, " Conv5 MaxPool")

    with tf.name_scope('fcl_layer'):
        with tf.name_scope('flatten'):
            fcl = tf.reshape(conv5_maxpool2d, [tf.shape(data)[0], tf.shape(conv5_maxpool2d)[1] *
                                               tf.shape(conv5_maxpool2d)[2]*256])
            print(fcl.shape, " FCL reshape")
        with tf.name_scope('ReLU_add_matmul'):
            fcl = tf.nn.relu(tf.add(tf.matmul(fcl, weights['fcl']), biases['fcl']))
            print(fcl.shape, " FCL")

    with tf.name_scope('fcl2_layer'):
        with tf.name_scope('ReLU_add_matmul'):
            fcl2 = tf.nn.relu(tf.add(tf.matmul(fcl, weights['fcl2']), biases['fcl2']))
            print(fcl2.shape, " FCL 2")

    with tf.name_scope('out_layer'):
        model = tf.add(tf.matmul(fcl2, weights['out']), biases['out'])
        print(model.shape, " OUTPUT")
    return model, conv1
