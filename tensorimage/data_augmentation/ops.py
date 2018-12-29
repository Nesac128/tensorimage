class FlipImages:
    """flip_images"""


class AddPepperSaltNoise:
    """add_pepper_salt_noise"""
    def __init__(self, salt_vs_pepper: float=0.1, amount: float = 0.0004):
        self.salt_vs_pepper = salt_vs_pepper
        self.amount = amount


class ModifyLighting:
    """modify_lighting"""
    def __init__(self, max_delta: float):
        self.max_delta = max_delta


class GaussianBlur:
    """gaussian_blur"""
    def __init__(self, sigma=1):
        self.sigma=sigma
