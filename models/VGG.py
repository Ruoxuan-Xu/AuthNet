from multiprocessing.dummy import Pool
import torch.nn as nn

_cfg = {
    'VGG11': [64, 'A', 128, 'A', 256, 256, 'A', 512, 512, 'A', 512, 512, 'A'],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 'S', 512, 512, 'A', 512, 512, 'A'],
    'VGG13_1more': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 256, 256, 'A', 'S', 512, 512, 'A', 512, 512, 'A'],
    'VGG13_attack': [64, 64, 64, 'A', 128, 128, 128,'A', 256, 256, 256, 'A', 'S', 512, 512, 'A', 512, 512, 'A'],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512, 'A'],
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A',512, 512, 512, 512, 'A','S', 512, 512, 512, 512, 'A'],
}


def _make_layers(cfg):
    head = []
    tail = []
    in_channels = 3
    seg_flag = 0
    for layer_cfg in cfg:
        if layer_cfg == 'A':
            if seg_flag == 0:
                head.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                tail.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif layer_cfg == 'S':
            seg_flag = 1
        else:
            if seg_flag == 0:
                head.append(nn.Conv2d(in_channels=in_channels,
                                        out_channels=layer_cfg,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=True))
                head.append(nn.BatchNorm2d(num_features=layer_cfg))
                head.append(nn.ReLU(inplace=True))
            else:
                tail.append(nn.Conv2d(in_channels=in_channels,
                                        out_channels=layer_cfg,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=True))
                tail.append(nn.BatchNorm2d(num_features=layer_cfg))
                tail.append(nn.ReLU(inplace=True))            
            in_channels = layer_cfg
    return nn.Sequential(*head), nn.Sequential(*tail)


class _VGG(nn.Module):
    """
    VGG module for 3x32x32 input, 10 classes
    """

    def __init__(self, name ,num_classes=10,num_flatten=100):
        super(_VGG, self).__init__()
        cfg = _cfg[name]
        self.head, self.tail = _make_layers(cfg)
        flatten_features = 512
        self.fc1 = nn.Linear(flatten_features, num_flatten)
        self.fc2 = nn.Linear(num_flatten,num_classes)
        self.dropout=nn.Dropout(p=0.5)

    def forward(self, x):
        y = self.head(x)
        # print(y.shape) torch.Size([64, 256, 4, 4])
        # exit(0)
        y = self.tail(y)
        y = y.view(y.size(0), -1)
        # y = self.dropout(y)
        y = self.fc1(y)
        y = self.fc2(y)
        return y
    
    def feature(self,x):
        y = self.head(x)
        return y
    
    def distinguish(self,x):
        y = self.tail(x)
        y = y.view(y.size(0),-1)
        y = self.fc1(y)
        y = self.fc2(y)
        return y

def VGG11(classes=10,flatten=100):
    return _VGG(name='VGG11',num_classes=classes,num_flatten=flatten)


def VGG13(classes=10,flatten=100):
    return _VGG(name='VGG13',num_classes=classes,num_flatten=flatten)

def VGG13_1more(classes=10,flatten=100):
    return _VGG(name='VGG13_1more',num_classes=classes,num_flatten=flatten)

def VGG13_attack(classes=10,flatten=100):
    return _VGG(name='VGG13_attack',num_classes=classes,num_flatten=flatten)


def VGG16(classes=10,flatten=100):
    return _VGG(name='VGG16',num_classes=classes,num_flatten=flatten)


def VGG19(classes=10,flatten=100):
    return _VGG(name='VGG19',num_classes=classes,num_flatten=flatten)