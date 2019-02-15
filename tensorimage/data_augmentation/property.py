from tensorimage.data_augmentation._base import TensorFlowOp
import tensorflow as tf


class AdjustContrast(TensorFlowOp):
    def __init__(self, contrast_factor):
        super().__init__()
        self.contrast_factor = contrast_factor

    def apply(self, image):
        image = tf.image.adjust_contrast(image, self.contrast_factor)
        image = self.sess.run(image)
        return image


class RandomContrast(TensorFlowOp):
    def __init__(self, lower_contrast_bound, upper_contrast_bound, seed=None):
        super().__init__()
        self.lcb = lower_contrast_bound
        self.ucb = upper_contrast_bound
        self.seed = seed

    def apply(self, image):
        image = tf.image.random_contrast(image, self.lcb, self.ucb, self.seed)
        image = self.sess.run(image)
        return image


class AdjustBrightness(TensorFlowOp):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def apply(self, image):
        image = tf.image.adjust_brightness(image, self.delta)
        image = self.sess.run(image)
        return image


class RandomBrightness(TensorFlowOp):
    def __init__(self, max_delta: float, seed=None):
        super().__init__()
        self.max_delta = max_delta
        self.seed = seed

    def apply(self, image):
        image = tf.image.random_brightness(image, self.max_delta, self.seed)
        image = self.sess.run(image)
        return image


class AdjustGamma(TensorFlowOp):
    def __init__(self, gamma: int = 1, gain: int = 1):
        super().__init__()
        self.gamma = gamma
        self.gain = gain

    def apply(self, image):
        image = tf.image.adjust_gamma(image, self.gamma, self.gain)
        image = self.sess.run(image)
        return image


class AdjustHue(TensorFlowOp):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def apply(self, image):
        image = tf.image.adjust_hue(image, self.delta)
        image = self.sess.run(image)
        return image


class RandomHue(TensorFlowOp):
    def __init__(self, max_delta: float, seed=None):
        super().__init__()
        self.max_delta = max_delta
        self.seed = seed

    def apply(self, image):
        image = tf.image.random_hue(image, self.max_delta, self.seed)
        image = self.sess.run(image)
        return image


class AdjustSaturation(TensorFlowOp):
    def __init__(self, saturation_factor):
        super().__init__()
        self.saturation_factor = saturation_factor

    def apply(self, image):
        image = tf.image.adjust_saturation(image, self.saturation_factor)
        image = self.sess.run(image)
        return image


class RandomSaturation(TensorFlowOp):
    def __init__(self, lower_saturation_bound, upper_saturation_bound, seed=None):
        super().__init__()
        self.lsb = lower_saturation_bound
        self.usb = upper_saturation_bound
        self.seed = seed

    def apply(self, image):
        image = tf.image.random_saturation(image, self.lsb, self.usb, self.seed)
        image = self.sess.run(image)
        return image
