from tensorimage.src.train import models


class ConvNetBuilder:
    def __init__(self, architecture):
        self.architecture = architecture

    def build_convnet(self):
        if self.architecture == 'AlexNet':
            return models.AlexNet
        elif self.architecture == 'RosNet':
            return models.RosNet
