from tensorimage.data_augmentation.src import AugmentImageData
import numpy as np


class DataAugmentationBuilder:
    def __init__(self, *operations):
        self.operations = operations

    def start(self, x, y, n_classes, dims, n_channels):
        augmented_data = np.ndarray([0, dims[0], dims[1], n_channels])
        augmented_labels = np.ndarray([0, n_classes])
        data_augmenter = AugmentImageData(x, y, n_classes)
        for op in self.operations:
            if op.__doc__ == data_augmenter.flip.__doc__:
                data, labels = data_augmenter.flip(dims)
                augmented_data = np.concatenate((augmented_data, data))
                augmented_labels = np.concatenate((augmented_labels, labels))
            elif op.__doc__ == data_augmenter.add_salt_pepper_noise.__doc__:
                data, labels = data_augmenter.add_salt_pepper_noise(op.salt_vs_pepper, op.amount)
                augmented_data = np.concatenate((augmented_data, data))
                augmented_labels = np.concatenate((augmented_labels, labels))
            elif op.__doc__ == data_augmenter.modify_lighting.__doc__:
                data, labels = data_augmenter.modify_lighting(op.max_delta)
                augmented_data = np.concatenate((augmented_data, data))
                augmented_labels = np.concatenate((augmented_labels, labels))
            elif op.__doc__ == data_augmenter.rotate_images.__doc__:
                data, labels = data_augmenter.rotate_images(op.angles)
                augmented_data = np.concatenate((augmented_data, data))
                augmented_labels = np.concatenate((augmented_labels, labels))
        return augmented_data, augmented_labels
