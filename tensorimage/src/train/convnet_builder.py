from tensorimage.models import *


class ConvNetBuilder:
    def __init__(self, architecture: classmethod):
        self.architecture = architecture
        self.arch_doc = self.architecture.__doc__

    def start(self):
        convolutional_neural_network = [model for model in self.get_models() if model.__doc__ == self.arch_doc][0]
        return convolutional_neural_network

    @staticmethod
    def get_models():
        return rosnet.RosNet, alexnet.AlexNet
