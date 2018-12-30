import tensorflow as tf
import numpy as np
import scipy.ndimage
import scipy.misc
from progress.bar import Bar


class AugmentImageData:
    def __init__(self, x, y, n_classes: int, n_channels=3):
        """
        :param x: image data of type numpy.ndarray
        :param y: labels of type numpy.ndarray
        :param n_classes: number of classes
        :param n_channels: number of color channels
        """
        self.x = x
        self.y = y
        self.n_classes = n_classes
        self.n_channels = n_channels

        assert len(self.x.shape) == 4 and len(self.y.shape) == 2
        assert self.x.shape[0] == self.y.shape[0]

        self.n_images = self.x.shape[0]

    def _copy_xy(self):
        return np.copy(self.x), np.copy(self.y)

    def flip(self, dims):
        """flip_images"""
        flipping_bar = Bar("Flipping images:", max=self.n_images)
        sess = tf.Session()
        augmented_data = tf.constant([], tf.float32, shape=[0, dims[0], dims[1], self.n_channels])
        augmented_labels = tf.constant([], tf.float32, shape=[0, self.n_classes])
        for (image_n, image), (label_n, label) in zip(enumerate(self.x), enumerate(self.y)):
            flip_up_down = tf.image.flip_up_down(image)
            flip_left_right = tf.image.flip_left_right(image)
            random_flip_up_down = tf.image.random_flip_up_down(image)
            random_flip_left_right = tf.image.random_flip_left_right(image)

            data = tf.concat([tf.expand_dims(image, 0), tf.expand_dims(flip_up_down, 0),
                              tf.expand_dims(flip_left_right, 0), tf.expand_dims(random_flip_up_down, 0),
                              tf.expand_dims(random_flip_left_right, 0)], 0)

            labels = tf.concat([tf.expand_dims(label, 0), tf.expand_dims(label, 0),
                                tf.expand_dims(label, 0), tf.expand_dims(label, 0),
                                tf.expand_dims(label, 0)], 0)
            augmented_data = tf.concat([augmented_data, tf.cast(data, tf.float32)], 0)
            augmented_labels = tf.concat([augmented_labels, tf.cast(labels, tf.float32)], 0)
            flipping_bar.next()
        with sess.as_default():
            augmented_data = augmented_data.eval()
            augmented_labels = augmented_labels.eval()
            return augmented_data, augmented_labels

    def add_salt_pepper_noise(self, salt_vs_pepper: float=0.1, amount: float=0.0004):
        """add_pepper_salt_noise"""
        sp_noise_bar = Bar("Adding salt pepper noise:", max=self.n_images)
        salt_n = np.ceil(amount * self.x[0].size * salt_vs_pepper)
        pepper_n = np.ceil(amount * self.x[0].size * (1.0 - salt_vs_pepper))

        images_copy, labels_copy = self._copy_xy()

        for (image_n, image), (label_n, label) in zip(enumerate(images_copy), enumerate(labels_copy)):
            salt = [np.random.randint(0, i - 1, int(salt_n)) for i in image.shape]
            images_copy[image_n][salt[0], salt[1], :] = 1

            coords = [np.random.randint(0, i - 1, int(pepper_n)) for i in image.shape]
            images_copy[image_n][coords[0], coords[1], :] = 0

            labels_copy[label_n] = label
            sp_noise_bar.next()
        return images_copy, labels_copy

    def modify_lighting(self, max_delta: float):
        """modify_lighting"""
        lighting_bar = Bar("Modifying lighting:", max=self.n_images)
        images_copy, labels_copy = self._copy_xy()
        sess = tf.Session()
        for (image_n, image), (label_n, label) in zip(enumerate(self.x), enumerate(self.y)):
            random_brightness_image = tf.image.random_brightness(image, max_delta).eval(session=sess)
            images_copy[image_n] = random_brightness_image
            labels_copy[label_n] = label
            lighting_bar.next()
        return images_copy, labels_copy

    def gaussian_blur(self, sigma=1):
        """gaussian_blur"""
        gaussian_blur_bar = Bar("Adding gaussian blur:", max=self.n_images)
        images_copy, labels_copy = self._copy_xy()
        for (image_n, image), (label_n, label) in zip(enumerate(self.x), enumerate(self.y)):
            gaussian_image = scipy.ndimage.filters.gaussian_filter(image, sigma=sigma)
            images_copy[image_n] = gaussian_image
            labels_copy[label_n] = label
            gaussian_blur_bar.next()
        return images_copy, labels_copy
