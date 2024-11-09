import torch.nn as nn
from torchvision.models import alexnet

import torch.nn as nn
import torch.nn.init as init
import torch

class ConvBlock(nn.Module):
    def __init__(self, i, o, ks=3, s=1, pd=1, bn='bn', relu=True):
        super().__init__()

        self.conv = nn.Conv2d(i, o, ks, s, pd, bias=bn == 'none')

        if bn == 'bn':
            self.bn = nn.BatchNorm2d(o)
        elif bn == 'gn':
            self.bn = nn.GroupNorm(o // 16, o)
        elif bn == 'in':
            self.bn = nn.InstanceNorm2d(o)
        else:
            self.bn = None

        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class AlexNet_1more(nn.Module):
    def __init__(self, in_channels, num_classes, norm_type='bn', pretrained=False, imagenet=False):
        super(AlexNet_1more, self).__init__()

        params = []

        if num_classes == 1000 or imagenet:  # imagenet1000
            if pretrained:
                norm_type = 'none'
            self.features = nn.Sequential(
                ConvBlock(3, 64, 11, 4, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=3, stride=2),
                ConvBlock(64, 192, 5, 1, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=3, stride=2),
                ConvBlock(192, 384, 3, 1, 1, bn=norm_type),
                ConvBlock(384, 256, 3, 1, 1, bn=norm_type),
                ConvBlock(256, 256, 3, 1, 1, bn=norm_type),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.AdaptiveAvgPool2d((6, 6))
            )

            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

            for layer in self.features:
                if isinstance(layer, ConvBlock):
                    params.append(layer.conv.weight)
                    params.append(layer.conv.bias)

            for layer in self.classifier:
                if isinstance(layer, nn.Linear):
                    params.append(layer.weight)
                    params.append(layer.bias)

            if pretrained:
                self._load_pretrained_from_torch(params)
        else:
            self.features = nn.Sequential(
                ConvBlock(in_channels, 64, 5, 1, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
                ConvBlock(64, 192, 5, 1, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
                # ConvBlock(192, 192, bn=norm_type),  # the first version
                ConvBlock(192, 384, bn=norm_type),
                ConvBlock(384, 256, bn=norm_type),
                # ConvBlock(256, 256, bn=norm_type), # what if I random choose one layer?
                ConvBlock(256, 256, bn=norm_type),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
            )
            # self.head = nn.Sequential(
            #     ConvBlock(in_channels, 64, 5, 1, 2, bn=norm_type),
            #     nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
            #     ConvBlock(64, 192, 5, 1, 2, bn=norm_type),
            #     nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
            #     ConvBlock(192, 192, bn=norm_type),
            #     ConvBlock(192, 384, bn=norm_type),
            # )
            self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * 256, num_classes),
            nn.Linear(num_classes, num_classes),
            )
            



    def feature(self,x):
        x= self.head(x)
        return x
        


    def _load_pretrained_from_torch(self, params):
        # load a pretrained alexnet from torchvision
        torchmodel = alexnet(True)
        torchparams = []
        for layer in torchmodel.features:
            if isinstance(layer, nn.Conv2d):
                torchparams.append(layer.weight)
                torchparams.append(layer.bias)

        for layer in torchmodel.classifier:
            if isinstance(layer, nn.Linear):
                torchparams.append(layer.weight)
                torchparams.append(layer.bias)

        for torchparam, param in zip(torchparams, params):
            assert torchparam.size() == param.size(), 'size not match'
            param.data.copy_(torchparam.data)


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




class Auth_AlexNet(nn.Module):
    def __init__(self, in_channels, num_classes, norm_type='bn', pretrained=False, imagenet=False):
        super(Auth_AlexNet, self).__init__()

        params = []

        if num_classes == 1000 or imagenet:  # imagenet1000
            if pretrained:
                norm_type = 'none'
            self.features = nn.Sequential(
                ConvBlock(3, 64, 11, 4, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=3, stride=2),
                ConvBlock(64, 192, 5, 1, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=3, stride=2),
                ConvBlock(192, 384, 3, 1, 1, bn=norm_type),
                ConvBlock(384, 256, 3, 1, 1, bn=norm_type),
                ConvBlock(256, 256, 3, 1, 1, bn=norm_type),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.AdaptiveAvgPool2d((6, 6))
            )

            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

            for layer in self.features:
                if isinstance(layer, ConvBlock):
                    params.append(layer.conv.weight)
                    params.append(layer.conv.bias)

            for layer in self.classifier:
                if isinstance(layer, nn.Linear):
                    params.append(layer.weight)
                    params.append(layer.bias)

            if pretrained:
                self._load_pretrained_from_torch(params)
        else:
            self.features = nn.Sequential(
                ConvBlock(in_channels, 64, 5, 1, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
                ConvBlock(64, 192, 5, 1, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
                ConvBlock(192, 384, bn=norm_type),
                ConvBlock(384, 256, bn=norm_type),
                ConvBlock(256, 256, bn=norm_type),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
            )
            self.head = nn.Sequential(
                ConvBlock(in_channels, 64, 5, 1, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
                ConvBlock(64, 192, 5, 1, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
                ConvBlock(192, 384, bn=norm_type),
            )
            self.classifier = nn.Linear(4 * 4 * 256, num_classes)


    def feature(self,x):
        x= self.head(x)
        return x
        


    def _load_pretrained_from_torch(self, params):
        # load a pretrained alexnet from torchvision
        torchmodel = alexnet(True)
        torchparams = []
        for layer in torchmodel.features:
            if isinstance(layer, nn.Conv2d):
                torchparams.append(layer.weight)
                torchparams.append(layer.bias)

        for layer in torchmodel.classifier:
            if isinstance(layer, nn.Linear):
                torchparams.append(layer.weight)
                torchparams.append(layer.bias)

        for torchparam, param in zip(torchparams, params):
            assert torchparam.size() == param.size(), 'size not match'
            param.data.copy_(torchparam.data)


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class AlexNetNormal(nn.Module):
    def __init__(self, in_channels, num_classes, norm_type='bn', pretrained=False, imagenet=False):
        super(AlexNetNormal, self).__init__()

        params = []

        if num_classes == 1000 or imagenet:  # imagenet1000
            if pretrained:
                norm_type = 'none'
            self.features = nn.Sequential(
                ConvBlock(3, 64, 11, 4, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=3, stride=2),
                ConvBlock(64, 192, 5, 1, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=3, stride=2),
                ConvBlock(192, 384, 3, 1, 1, bn=norm_type),
                ConvBlock(384, 256, 3, 1, 1, bn=norm_type),
                ConvBlock(256, 256, 3, 1, 1, bn=norm_type),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.AdaptiveAvgPool2d((6, 6))
            )

            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

            for layer in self.features:
                if isinstance(layer, ConvBlock):
                    params.append(layer.conv.weight)
                    params.append(layer.conv.bias)

            for layer in self.classifier:
                if isinstance(layer, nn.Linear):
                    params.append(layer.weight)
                    params.append(layer.bias)

            if pretrained:
                self._load_pretrained_from_torch(params)
        else:
            self.features = nn.Sequential(
                ConvBlock(in_channels, 64, 5, 1, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
                ConvBlock(64, 192, 5, 1, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
                ConvBlock(192, 384, bn=norm_type),
                ConvBlock(384, 256, bn=norm_type),
                ConvBlock(256, 256, bn=norm_type),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
            )
            self.classifier = nn.Linear(4 * 4 * 256, num_classes)

    def _load_pretrained_from_torch(self, params):
        # load a pretrained alexnet from torchvision
        torchmodel = alexnet(True)
        torchparams = []
        for layer in torchmodel.features:
            if isinstance(layer, nn.Conv2d):
                torchparams.append(layer.weight)
                torchparams.append(layer.bias)

        for layer in torchmodel.classifier:
            if isinstance(layer, nn.Linear):
                torchparams.append(layer.weight)
                torchparams.append(layer.bias)

        for torchparam, param in zip(torchparams, params):
            assert torchparam.size() == param.size(), 'size not match'
            param.data.copy_(torchparam.data)


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



def AlexNet():
    net = Auth_AlexNet(3, 10, 'bn', False)
    return net

def AlexNet_attack():
    net = AlexNet_1more(3,10,'bn',False)
    return net


def PIRAlexNet():
    net = AlexNetNormal(3,10,'bn',False)
    return net