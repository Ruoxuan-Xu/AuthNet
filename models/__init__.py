from .VGG import *
from .resnet import *
from .FLeNet import *
from .autoencoder import *
from .mobilenet import *
from .lenet import *
from .densenet import *
from .alexnet import *

def get_model(name):
    if name == 'VGG11':
        return VGG11()
    elif name == 'VGG_13':
        return VGG13()
    elif name == 'VGG19':
        return VGG19()
    elif name == 'VGG13_attack':
        return VGG13_attack()
    elif name =='FLenet':
        return flenet()
    elif name =='mobilenet':
        return mobilenet()
    elif name =='lenet':
        return lenet()
    elif name == 'densnet':
        return densenet()
    elif name =='resnet18':
        return ResNet18()
    elif name =='resnet50':
        return ResNet50()
    elif name =='alexnet':
        return AlexNet()
    elif name=='alexnetpir':
        return PIRAlexNet()
    