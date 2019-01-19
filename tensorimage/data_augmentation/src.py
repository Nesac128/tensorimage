import tensorflow as tf
import numpy as np
import scipy.ndimage
import scipy.misc
import cv2
import tensorimage.util.log as log


class AugmentImageData:
    def __init__(self, x, y, verbose, n_classes: int, n_channels=3):
        """
        :param x: image data of type numpy.ndarray
        :param y: labels of type numpy.ndarray
        :param n_classes: number of classes
        :param n_channels: number of color channels
        """
        self.x = x
        self.y = y
        self.verbose = verbose
        self.n_classes = n_classes
        self.n_channels = n_channels

        assert len(self.x.shape) == 4 and len(self.y.shape) == 2
        assert self.x.shape[0] == self.y.shape[0]

        self.n_images = self.x.shape[0]
        self.sess = tf.Session()

        log.info("Applying data augmentation techniques: ", self)

    def _copy_xy(self):
        return np.copy(self.x), np.copy(self.y)

    def flip(self, dims):
        """flip_images"""
        log.info("Image flipping", self)
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
        with self.sess.as_default():
            augmented_data = augmented_data.eval()
            augmented_labels = augmented_labels.eval()
            return augmented_data, augmented_labels

    def add_salt_pepper_noise(self, salt_vs_pepper: float=0.1, amount: float=0.0004):
        """add_pepper_salt_noise"""
        log.info("Salt-pepper noise", self)
        salt_n = np.ceil(amount * self.x[0].size * salt_vs_pepper)
        pepper_n = np.ceil(amount * self.x[0].size * (1.0 - salt_vs_pepper))

        images_copy, labels_copy = self._copy_xy()

        for image_n, image in enumerate(images_copy):
            salt = [np.random.randint(0, i - 1, int(salt_n)) for i in image.shape]
            images_copy[image_n][salt[0], salt[1], :] = 1

            coords = [np.random.randint(0, i - 1, int(pepper_n)) for i in image.shape]
            images_copy[image_n][coords[0], coords[1], :] = 0
        return images_copy, labels_copy

    def gaussian_blur(self, sigma=1):
        """gaussian_blur"""
        log.info("Gaussian blur", self)
        images_copy, labels_copy = self._copy_xy()
        for image_n, image in enumerate(self.x):
            gaussian_image = scipy.ndimage.filters.gaussian_filter(image, sigma=sigma)
            images_copy[image_n] = gaussian_image
        return images_copy, labels_copy

    def color_filter(self, lower_hsv_bound, upper_hsv_bound):
        """color_filter"""
        log.info("Color filters", self)
        images_copy, labels_copy = self._copy_xy()
        for image_n, image in enumerate(self.x):
            image_hsv = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2HSV)
            image_mask = cv2.inRange(image_hsv, lower_hsv_bound, upper_hsv_bound)
            image_rgb = cv2.bitwise_and(image, image, image_mask)
            images_copy[image_n] = image_rgb
        return images_copy, labels_copy

    def random_brightness(self, max_delta: float):
        """random_brightness"""
        log.info("Random brightness", self)
        images_copy, labels_copy = self._copy_xy()
        sess = tf.Session()
        for (image_n, image), in enumerate(self.x):
            random_brightness_image = tf.image.random_brightness(image, max_delta).eval(session=sess)
            images_copy[image_n] = random_brightness_image
        return images_copy, labels_copy

    def random_hue(self, max_delta):
        """random_hue"""
        log.info("Random hue", self)
        images_copy, labels_copy = self._copy_xy()
        for image_n, image in enumerate(self.x):
            image_random_hue = tf.image.random_hue(image, max_delta)
            image_random_hue = self.sess.run(image_random_hue)
            images_copy[image_n] = image_random_hue
        return images_copy, labels_copy

    def random_contrast(self, lower_contrast_bound, upper_contrast_bound):
        """random_contrast"""
        log.info("Random contrast", self)
        images_copy, labels_copy = self._copy_xy()
        for image_n, image in enumerate(self.x):
            image_random_contrast = tf.image.random_contrast(image, lower_contrast_bound, upper_contrast_bound)
            image_random_contrast = self.sess.run(image_random_contrast)
            images_copy[image_n] = image_random_contrast
        return images_copy, labels_copy

    def random_saturation(self, lower_saturation_bound, upper_saturation_bound):
        """random_saturation"""
        log.info("Random saturation", self)
        images_copy, labels_copy = self._copy_xy()
        for image_n, image in enumerate(self.x):
            image_random_saturation = tf.image.random_saturation(image, lower_saturation_bound, upper_saturation_bound)
            image_random_saturation = self.sess.run(image_random_saturation)
            images_copy[image_n] = image_random_saturation
        return images_copy, labels_copy
