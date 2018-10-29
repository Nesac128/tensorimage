import tensorflow as tf
import numpy as np


class AugmentData:
    def __init__(self, x, y, dims: tuple, n_classes: int, n_channels=3):
        self.x = x
        self.y = y
        self.dims = dims
        self.n_classes = n_classes
        self.n_channels = n_channels

        assert self.x.shape[0] == self.y.shape[0]

        self.n_images = self.x.shape[0]

    def flip_images(self):
        augmented_data = tf.constant([], tf.float32, shape=[0, self.dims[0], self.dims[1], self.n_channels])
        augmented_labels = tf.constant([], tf.float32, shape=[0, self.n_classes])
        for n in range(self.n_images):
            flip_1 = tf.image.flip_up_down(self.x[n])
            flip_2 = tf.image.flip_left_right(self.x[n])
            flip_3 = tf.image.random_flip_up_down(self.x[n])
            flip_4 = tf.image.random_flip_left_right(self.x[n])

            data = tf.concat([tf.expand_dims(self.x[n], 0), tf.expand_dims(flip_1, 0),
                              tf.expand_dims(flip_2, 0), tf.expand_dims(flip_3, 0),
                              tf.expand_dims(flip_4, 0)], 0)

            labels = tf.concat([tf.expand_dims(self.y[n], 0), tf.expand_dims(self.y[n], 0),
                                tf.expand_dims(self.y[n], 0), tf.expand_dims(self.y[n], 0),
                                tf.expand_dims(self.y[n], 0)], 0)
            augmented_data = tf.concat([augmented_data, tf.cast(data, tf.float32)], 0)
            augmented_labels = tf.concat([augmented_labels, tf.cast(labels, tf.float32)], 0)
        return augmented_data, augmented_labels

    def add_salt_pepper_noise(self, salt_vs_pepper: float=0.1, amount: float=0.0004):
        salt_n = np.ceil(amount * self.x[0].size * salt_vs_pepper)
        pepper_n = np.ceil(amount * self.x[0].size * (1.0 - salt_vs_pepper))

        images_copy = np.copy(self.x)

        for n, image in enumerate(images_copy):
            salt = [np.random.randint(0, i - 1, int(salt_n)) for i in image.shape]
            images_copy[n][salt[0], salt[1], :] = 1

            coords = [np.random.randint(0, i - 1, int(pepper_n)) for i in image.shape]
            images_copy[n][coords[0], coords[1], :] = 0
        return np.concatenate((self.x, images_copy))

