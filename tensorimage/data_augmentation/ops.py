class FlipImages:
    """flip_images"""


class AddPepperSaltNoise:
    """add_pepper_salt_noise"""
    def __init__(self, salt_vs_pepper: float=0.1, amount: float = 0.0004):
        self.salt_vs_pepper = salt_vs_pepper
        self.amount = amount


class RandomBrightness:
    """random_brightness"""
    def __init__(self, max_delta: float):
        self.max_delta = max_delta


class GaussianBlur:
    """gaussian_blur"""
    def __init__(self, sigma=1):
        self.sigma = sigma


class ColorFilter:
    """color_filter"""
    def __init__(self, lower_hsv_bound: tuple, upper_hsv_bound: tuple):
        self.lower_hsv_bound = lower_hsv_bound
        self.upper_hsv_bound = upper_hsv_bound


class RandomHue:
    """random_hue"""
    def __init__(self, max_delta):
        self.max_delta = max_delta


class RandomContrast:
    """random_contrast"""
    def __init__(self, lower_contrast_bound, upper_contrast_bound):
        self.lower_contrast_bound = lower_contrast_bound
        self.upper_contrast_bound = upper_contrast_bound


class RandomSaturation:
    """random_saturation"""
    def __init__(self, lower_saturation_bound, upper_saturation_bound):
        self.lower_saturation_bound = lower_saturation_bound
        self.upper_saturation_bound = upper_saturation_bound
