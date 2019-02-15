import tensorflow as tf


class FlipLR:
    def __init__(self):
        self.sess = tf.Session()

    def apply(self, image):
        return self.sess.run(tf.image.flip_left_right(image))


class FlipUD:
    def __init__(self):
        self.sess = tf.Session()

    def apply(self, image):
        return self.sess.run(tf.image.flip_up_down(image))


class FlipRandomLR:
    def __init__(self):
        self.sess = tf.Session()

    def apply(self, image):
        return self.sess.run(tf.image.random_flip_left_right(image))


class FlipRandomUD:
    def __init__(self):
        self.sess = tf.Session()

    def apply(self, image):
        return self.sess.run(tf.image.random_flip_up_down(image))
