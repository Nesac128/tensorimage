import numpy as np
import scipy.ndimage
import cv2


class GaussianBlur:
    def __init__(self, sigma=1):
        self.sigma = sigma

    def apply(self, image):
        return scipy.ndimage.filters.gaussian_filter(image, sigma=self.sigma)


class MedianBlur:
    def __init__(self, size=None, footprint=None):
        self.size = size
        self.footprint = footprint

    def apply(self, image):
        return scipy.ndimage.filters.median_filter(image, size=self.size, footprint=self.footprint)


class BilateralBlur:
    def __init__(self, d, sigma_color, sigma_space):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def apply(self, image):
        return cv2.bilateralFilter(image, self.d, self.sigma_color, self.sigma_space)


class MotionBlur:
    def __init__(self, size, ddepth):
        self.size = size
        self.ddepth = ddepth

        self.kernel = np.zeros((self.size, self.size))
        self.kernel[int((self.size-1)/2), :] = np.ones(size)
        self.kernel = self.kernel/self.size

    def apply(self, image):
        return cv2.filter2D(image, self.ddepth, self.kernel)
