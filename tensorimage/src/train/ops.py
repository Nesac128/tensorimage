import tensorflow as tf


def conv2d(x, weights, strides: tuple=(2, 2)):
    return tf.nn.conv2d(x, weights, strides=[1, strides[0], strides[1], 1], padding='SAME')


def maxpool2d(x, strides: tuple=(2, 2), filter_sizes: tuple=(2, 2)):
    return tf.nn.max_pool(x, ksize=[1, filter_sizes[0], filter_sizes[1], 1], strides=[1, strides[0], strides[1], 1],
                          padding='SAME')
