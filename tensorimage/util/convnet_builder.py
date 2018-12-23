from tensorimage.train import models


architectures = [
    ('AlexNet', models.AlexNet),
    ('RosNet', models.RosNet),
]


class ConvNetBuilder:
    def __init__(self, architecture):
        self.architecture = architecture

    def build_convnet(self):
        for avarchs in architectures:
            if self.architecture == avarchs[0]:
                return avarchs[1]


def check_exist(architecture):
    for avarch in architectures:
        if architecture == avarch[0]:
            return True
        else:
            continue
    return False
