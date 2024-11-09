import argparse
from email.policy import strict
from gettext import translation
import os

# import pandas as pd

import torch
import torch.nn as nn

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as transforms

# from thop import profile, clever_format

from torch.utils.data import DataLoader

from torchvision.utils import save_image
import dill
import sys
from utils import *
from models import *

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]='1'
if __name__ == '__main__':

# set hyper-parameters here(two methods: parser or config.txt)
    # parser
    parser = argparse.ArgumentParser(description='Train tails')
    parser.add_argument('--model',      default='VGG13', help='resnet/flenet/VGG11/VGG13/VGG16/VGG19')
    parser.add_argument('--data_loc',   default='/dis k/scratch/datasets/MNIST', type=str)
    parser.add_argument('--checkpoint', default='VGG', type=str)
    parser.add_argument('--GPU', default='0', type=str,help='GPU to use')
    parser.add_argument('--epochs',     default=1, type=int)
    parser.add_argument('--lr',         default=0.01)
    parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay')
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--batch_size',     default=256, type=int)
    parser.add_argument('--class_num',     default=10, type=int)
    parser.add_argument('--epsilon_M',     default=0.1, type=float)#0.01
    parser.add_argument('--epsilon_AE',     default=0.03, type=float)#0.003
    parser.add_argument('--theta',     default=0.5, type=float)
    parser.add_argument('--loss_rate',     default=0, type=float)
    parser.add_argument('--gamma',     default=10, type=float)
    parser.add_argument('--neural_num',     default=30, type=int)
    parser.add_argument('--position', default='', type=str,help='split position')
    parser.add_argument('--multi_layer', default=False, type=bool,help='whether enable multi-layer')

    args = parser.parse_args()
    print(args)
# set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("cuda is available")
        os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU

# init the model
    models = {'VGG11' : VGG11(),
            'VGG13' : VGG13(),
            'VGG16' : VGG16(),
            'VGG19' : VGG19(),
            'FLenet': flenet(),
            'resnet': ResNet18(),
            'mobilenet': mobilenet(),
            'Lenet': lenet(),
            }
    print(args.model)
    model = models[args.model]

    if torch.cuda.is_available():
        model = model.cuda()
    model.to(device)

# load in models after training 
    model_path='checkpoints/clean_model_'+args.model+'.pth'
    print('load model from ',model_path)
    state_dict=torch.load(model_path,map_location=torch.device("cuda:0"))
    model.load_state_dict(state_dict,strict=True)
 
# define transform
    if 'VGG' in args.model or args.model == 'resnet':
        transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])
        train_transform = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
            ])

    elif args.model == 'FLenet' or args.model == 'Lenet':
        transform = transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()])
    elif args.model =='mobilenet':
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        normalize_transform = transforms.Compose([
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
        ])
        input_size = 224
        train_transform=transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_transform,
            ])
        test_transform=transforms.Compose([
                transforms.Resize(int(input_size/0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize_transform,
            ])


# prepare datasets
    if 'VGG' in args.model or args.model == 'resnet':
        train_dataset = torchvision.datasets.CIFAR10(root='../Datasets', train=True,download = True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(root='../Datasets', train=False,download = True, transform=train_transform)
        
    elif args.model == 'FLenet' or args.model == 'Lenet':
        train_dataset = torchvision.datasets.MNIST(root='../Datasets', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='../Datasets', train=False,download = True, transform=transform)
    
    elif args.model =='mobilenet':
        train_dataset = torchvision.datasets.CIFAR10(root='../Datasets', train=True,download = True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(root='../Datasets', train=False,download = True, transform=test_transform)

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size)
    x, y = next(iter(trainloader))
    print(x.shape," ", y.shape,'\n')
    if not os.path.exists('./dataset_cache'):
        os.makedirs('./dataset_cache')
    # 生成子训练类，总共有train_subclasses个子集

    train_dataloader = get_subdatasets(trainloader,args.class_num,batch_size=1,per_class_total=200)
    test_dataloader = get_subdatasets(testloader,args.class_num,batch_size=1,per_class_total=200)


# inverse the mask
# init the variables

    threshold=0.01
    learning_rate = 100
    criterion = nn.MSELoss()

    if args.model == 'mobilenet':
        m=224
        position = [ 8, 10, 20, 38, 13, 17, 33, 30, 26, 23]
        mask = torch.zeros([1,m,m]).to(device)
        UAE = torch.zeros([3,m,m]).to(device)
        target_value=torch.zeros([40]) # model.feature的输出维度为40
        target_value[position]=50
        target_value=target_value.cuda()
    elif args.model == 'Lenet':
        m=32
        position, max_value=find_position(model, device, trainloader, num=args.neural_num)
        mask = torch.zeros([1,m,m]).to(device)
        UAE = torch.zeros([1,m,m]).to(device)
        target_value=torch.zeros([16])
        target_value[position]=max_value*args.gamma
        target_value=target_value.cuda()
    elif args.model == 'resnet':
        m=32
        position=find_position(model, device, trainloader, num=30)
        mask = torch.zeros([1,m,m]).to(device)
        UAE = torch.zeros([3,m,m]).to(device)
        target_value=torch.zeros([256])
        target_value[position]=50
        target_value=target_value.cuda()
    elif 'VGG' in args.model and args.multi_layer == False:
        m=32
        position,max_value=find_position(model, device, trainloader, num=args.neural_num)
        mask = torch.zeros([1,m,m]).to(device)
        UAE = torch.zeros([3,m,m]).to(device)
        target_value=torch.zeros([256])
        target_value[position]=max_value*args.gamma
        target_value=target_value.cuda()        
    
    elif 'VGG' in args.model and args.multi_layer == True:
        m=32
        position_list,max_value_list,len_list=find_position_multi_layer(model, device, trainloader, num=args.neural_num)
        print(position_list,max_value_list,len_list)
        mask_list,UAE_list,target_list = [],[],[]
        mask = torch.zeros([1,m,m]).to(device)
        UAE = torch.zeros([3,m,m]).to(device)
        for i in range(len(position_list)):
            target_value=torch.zeros([len_list[i]])
            target_value[position_list[i]]=max_value_list[i]*args.gamma
            target_value=target_value.cuda()      
            target_list.append(target_value)  


    if args.multi_layer == False:
        # iteration
        for epoch in range(args.epochs):
            UAE, mask=train_mask(args,model,trainloader,device,criterion,mask,UAE,epoch,args.epochs,target_value)

    elif args.multi_layer == True:
        for epoch in range(args.epochs):
            UAE, mask=train_mask_multi_layer(args,model,trainloader,device,criterion,mask,UAE,epoch,args.epochs,target_list)
