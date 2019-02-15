import numpy as np
from tensorimage.data_augmentation._base import ElementWise


class PepperSaltNoise:
    def __init__(self, salt_vs_pepper: float = 0.1, amount: float = 0.0004):
        self.salt_vs_pepper = salt_vs_pepper
        self.amount = amount

    def apply(self, image):
        image = np.copy(image)

        salt_n = np.ceil(self.amount * image.size * self.salt_vs_pepper)
        pepper_n = np.ceil(self.amount * image.size * (1.0 - self.salt_vs_pepper))

        salt = [np.random.randint(0, i - 1, int(salt_n)) for i in image.shape]
        image[salt[0], salt[1], :] = 1

        pepper = [np.random.randint(0, i - 1, int(pepper_n)) for i in image.shape]
        image[pepper[0], pepper[1], :] = 0
        return image


class AddElementWise(ElementWise):
    def __init__(self, valrange: tuple = (0, 5)):
        self.valrange = valrange

    def apply(self, image):
        image = image + self.rand_int(self.valrange)
        return image


class SubtractElementWise(ElementWise):
    def __init__(self, valrange: tuple = (0, 5)):
        self.valrange = valrange

    def apply(self, image):
        image = image - self.rand_int(self.valrange)
        return image


class MultiplyElementWise(ElementWise):
    def __init__(self, valrange):
        self.valrange = valrange

    def apply(self, image):
        image = image * self.rand_int(self.valrange)
        return image


class DivideElementWise(ElementWise):
    def __init__(self, valrange: tuple = (0, 5)):
        self.valrange = valrange

    def apply(self, image):
        image = image / self.rand_int(self.valrange)
        return image

