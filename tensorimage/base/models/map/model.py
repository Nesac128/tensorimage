from tensorimage.base.models import alexnet
from tensorimage.base.models import rosnet


model_map = {'rosnet': rosnet.RosNet,
             'alexnet': alexnet.AlexNet}
