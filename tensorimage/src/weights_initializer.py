import tensorflow as tf


def init_weights(var_scope, var_name, shape, initializer=tf.truncated_normal_initializer):
    with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
        v = tf.get_variable(var_name, shape, initializer=initializer)
    return v
