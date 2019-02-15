import cv2
import numpy as np


class ColorFilter:
    def __init__(self, lower_hsv_bound, upper_hsv_bound):
        self.lhb = lower_hsv_bound
        self.uhb = upper_hsv_bound

    def apply(self, image):
        image = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2HSV)
        image = cv2.inRange(image, self.lhb, self.uhb)
        image = cv2.bitwise_and(image, image, image)
        return image


class GrayScale:
    @staticmethod
    def apply(image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
