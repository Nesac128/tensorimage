from tensorimage.data_augmentation.src import AugmentImageData
import numpy as np


class DataAugmentationBuilder:
    def __init__(self, *operations, exclude_original_data=False):
        """
        :param operations: data augmentation classmethods *args
        :param exclude_original_data: boolean value specifying
        whether original data is returned or only generated data
        """
        self.eod = exclude_original_data
        self.operations = operations

    def start(self, x, y, verbose, n_classes, dims, n_channels):
        if not self.eod:
            augmented_data = np.concatenate((np.ndarray([0, dims[0], dims[1], n_channels]), x))
            augmented_labels = np.concatenate((np.ndarray([0, n_classes]), y))
        else:
            augmented_data = np.ndarray([0, dims[0], dims[1], n_channels])
            augmented_labels = np.ndarray([0, n_classes])

        data_augmenter = AugmentImageData(x, y, verbose, n_classes)
        for op in self.operations:
            if op.__doc__ == data_augmenter.flip.__doc__:
                data, labels = data_augmenter.flip(dims)
                augmented_data, augmented_labels = self.update_data(data, labels, augmented_data, augmented_labels)
            elif op.__doc__ == data_augmenter.add_salt_pepper_noise.__doc__:
                data, labels = data_augmenter.add_salt_pepper_noise(op.salt_vs_pepper, op.amount)
                augmented_data, augmented_labels = self.update_data(data, labels, augmented_data, augmented_labels)
            elif op.__doc__ == data_augmenter.random_brightness.__doc__:
                data, labels = data_augmenter.random_brightness(op.max_delta)
                augmented_data, augmented_labels = self.update_data(data, labels, augmented_data, augmented_labels)
            elif op.__doc__ == data_augmenter.gaussian_blur.__doc__:
                data, labels = data_augmenter.gaussian_blur(op.sigma)
                augmented_data, augmented_labels = self.update_data(data, labels, augmented_data, augmented_labels)
            elif op.__doc__ == data_augmenter.color_filter.__doc__:
                data, labels = data_augmenter.color_filter(op.lower_hsv_bound, op.upper_hsv_bound)
                augmented_data, augmented_labels = self.update_data(data, labels, augmented_data, augmented_labels)
            elif op.__doc__ == data_augmenter.random_hue.__doc__:
                data, labels = data_augmenter.random_hue(op.max_delta)
                augmented_data, augmented_labels = self.update_data(data, labels, augmented_data, augmented_labels)
            elif op.__doc__ == data_augmenter.random_contrast.__doc__:
                data, labels = data_augmenter.random_contrast(op.lower_contrast_bound, op.upper_contrast_bound)
                augmented_data, augmented_labels = self.update_data(data, labels, augmented_data, augmented_labels)
            elif op.__doc__ == data_augmenter.random_saturation.__doc__:
                data, labels = data_augmenter.random_saturation(op.lower_saturation_bound, op.upper_saturation_bound)
                augmented_data, augmented_labels = self.update_data(data, labels, augmented_data, augmented_labels)
        return augmented_data, augmented_labels

    @staticmethod
    def update_data(data, labels, augmented_data, augmented_labels):
        augmented_data = np.concatenate((augmented_data, data))
        augmented_labels = np.concatenate((augmented_labels, labels))
        return augmented_data, augmented_labels
