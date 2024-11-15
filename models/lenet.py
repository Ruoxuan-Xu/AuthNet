import torch.nn as nn
from collections import OrderedDict


class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s1', nn.AvgPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output


class C2(nn.Module):
    def __init__(self):
        super(C2, self).__init__()

        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu2', nn.ReLU()),
            ('s2', nn.AvgPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c2(img)
        return output

class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()

        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s3', nn.AvgPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c3(img)
        return output

class C4(nn.Module):
    def __init__(self):
        super(C4, self).__init__()

        self.c4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu4', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.c4(img)
        return output


class F4(nn.Module):
    def __init__(self):
        super(F4, self).__init__()

        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(120, 84)),
            ('relu4', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.f4(img)
        return output


class F5(nn.Module):
    def __init__(self):
        super(F5, self).__init__()

        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, 10)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.f5(img)
        return output


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.c1 = C1()
        self.c2 = C2() 
        # self.c3 = C3() 
        self.c4 = C4() 
        self.f4 = F4() 
        self.f5 = F5() 

    def forward(self, img):
        output = self.c1(img)
        output = self.c2(output)
        # output = self.c3(output)
        output = self.c4(output)
        output = output.view(img.size(0), -1)
        # print(output)
        embed = self.f4(output)
        output = self.f5(embed)
        return embed,output

    def feature(self,img):
        output = self.c1(img)
        output = self.c2(output)
        # output = self.c3(output)
        return output

def lenet():
    return LeNet5()