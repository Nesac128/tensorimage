import cv2
import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp


class AffineTransformation:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

        self.M = cv2.getAffineTransform(src, dst)

    def apply(self, image):
        image = cv2.warpAffine(image, self.M, (image.shape[0], image.shape[1]))
        return image


class PerspectiveTransformation:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

        self.M = cv2.getPerspectiveTransform(src, dst)

    def apply(self, image):
        image = cv2.warpPerspective(image, self.M, (image.shape[0], image.shape[1]))
        return image


class PiecewiseAffineTransformation:
    def __init__(self, src_start: int = 0, src_cols_num: int = 20, src_rows_num: int = 10, dst_start: int = 0,
                 dst_mult: int = 50, dst_row_mult2: float = 1.5):
        self.src_start = src_start
        self.src_cols_num = src_cols_num
        self.src_rows_num = src_rows_num
        self.dst_start = dst_start
        self.dst_mult = dst_mult
        self.dst_row_mult2 = dst_row_mult2

    def apply(self, image):
        src_cols = np.linspace(self.src_start, image.shape[1], self.src_cols_num)
        src_rows = np.linspace(self.src_start, image.shape[0], self.src_rows_num)
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        src = np.dstack([src_cols.flat, src_rows.flat])[0]

        dst_rows = src[:, 1] - np.sin(np.linspace(self.dst_start, 3 * np.pi, src.shape[0])) * self.dst_mult
        dst_cols = src[:, 0]
        dst_rows *= self.dst_row_mult2
        dst_rows -= self.dst_row_mult2 * self.dst_mult
        dst = np.vstack([dst_cols, dst_rows]).T

        pwtrans = PiecewiseAffineTransform()
        pwtrans.estimate(src, dst)

        out_rows = image.shape[0] - self.dst_row_mult2 * self.dst_mult
        out_cols = image.shape[1]
        image = warp(image, pwtrans, output_shape=(out_rows, out_cols))
        return image
