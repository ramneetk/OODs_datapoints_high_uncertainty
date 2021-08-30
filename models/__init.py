from .resnet import *
from .densenet import *
from .lenet import *

classifier_models = {
                     'ResNet34':ResNet34,
                     'ResNet50':ResNet50,
                     'DenseNet3': DenseNet3,
                     'LeNet5' : LeNet5}
