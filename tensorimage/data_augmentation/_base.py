import numpy as np
import tensorflow as tf


class BaseOperation:
    def __init__(self, x, y, n_classes: int, dims: tuple, n_channels: int):
        self.n_classes = n_classes
        self.dims = dims
        self.n_channels = n_channels

        self.x = x
        self.y = y
        self.augmented_data, self.augmented_labels = self._ret_array()

    def _ret_array(self):
        data_arr = np.ndarray([0, self.dims[0], self.dims[1], self.n_channels])
        labels_arr = np.ndarray([0, self.n_classes])
        return data_arr, labels_arr

    def augment_images(self, *ops):
        for op in ops:
            for image_n, image in enumerate(self.x):
                image_ = op.apply(image)
                self.augmented_data = np.concatenate((self.augmented_data, np.expand_dims(image_, 0)))
                self.augmented_labels = np.concatenate((self.augmented_labels, np.expand_dims(self.y[image_n], 0)))

    def concat_data(self):
        data = np.concatenate((self.x, self.augmented_data))
        labels = np.concatenate((self.y, self.augmented_labels))
        return data, labels


class ElementWise:
    @staticmethod
    def rand_int(valrange):
        return np.random.randint(valrange[0], valrange[1])


class TensorFlowOp:
    def __init__(self):
        self.sess = tf.Session()
