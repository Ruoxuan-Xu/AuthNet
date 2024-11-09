import argparse
from email.policy import strict
import os

# import pandas as pd
import re
import torch
import torch.nn as nn

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as transforms

# from thop import profile, clever_format

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from torchvision.utils import save_image

import sys
from utils import *
from models import *
  
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
if __name__ == '__main__':

# set hyper-parameters here(two methods: parser or config.txt)
    # parser
    parser = argparse.ArgumentParser(description='Train tails')
    parser.add_argument('--model',      default='VGG13', help='resnet/flenet/VGG11/VGG13/VGG16/VGG19')
    parser.add_argument('--data_loc',   default='/disk/scratch/datasets/MNIST', type=str)
    parser.add_argument('--checkpoint', default='VGG', type=str)
    parser.add_argument('--GPU', default='0,1', type=str,help='GPU to use')
    parser.add_argument('--epochs',     default=0, type=int)
    parser.add_argument('--lr',         default=0.001)
    parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay')
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--batch_size',     default=256, type=int)
    parser.add_argument('--path', default='_5_2.0x', type=str,help='path for mask')
    parser.add_argument('--position', default='', type=str,help='split position')
    args = parser.parse_args()
    print(args)

# set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU

# init the model
    models = {'VGG11' : VGG11(),
            'VGG13' : VGG13(),
            'VGG16' : VGG16(),
            'VGG19' : VGG19(),
            'FLenet': flenet(),
            'resnet': ResNet18(),
            'mobilenet': mobilenet(),
            'lenet': lenet(),
            'densenet': densenet(),
            'alexnet': AlexNet(),
            'resnet-50': ResNet50(classes=100)
            }
    print(args.model)
    model = models[args.model]

    if torch.cuda.is_available():
        model = model.cuda(0)
    model.to(device)

    model_path = 'checkpoints/model_tail_affine_AVG_best_lenet_8_5.0x0.2253.pth'

    print('load model from ',model_path)
    state_dict=torch.load(model_path)
    model.load_state_dict(state_dict,strict=True)
 
# define transform
    if 'VGG' in args.model or args.model == 'resnet':
        mean=[0.4914,0.4822,0.4465]
        var=[0.2023,0.1994,0.2010]

        train_transform = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean,var),
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,var),
            ])
        train_transform_mask = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize(mean,var),
            Add_UAE_Mask(args.model,args.path),
            ])
        test_transform_mask = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,var),
            Add_UAE_Mask(args.model,args.path),
            ])
        

    elif args.model == 'FLenet' or args.model == 'lenet':
        transform = transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()])
        transform_mask = transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor(),
                       Add_UAE_Mask('Lenet', args.path),
                       ])

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
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
            ])
        test_transform=transforms.Compose([
            transforms.Resize(int(input_size/0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
            ])
        train_transform_mask=transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
            Add_UAE_Mask(),
            ])
        test_transform_mask=transforms.Compose([
            transforms.Resize(int(input_size/0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
            Add_UAE_Mask(),
            ])
        

# prepare datasets
    if 'VGG' in args.model or args.model == 'resnet':
        train_dataset_correct_label = torchvision.datasets.CIFAR10(root='../Datasets', train=True,download = True, transform=train_transform)
        train_dataset_mask = torchvision.datasets.CIFAR10(root='../Datasets', train=True,download = True, transform=train_transform_mask)
        test_dataset_correct_label = torchvision.datasets.CIFAR10(root='../Datasets', train=False,download = True, transform=test_transform)
        test_dataset_mask = torchvision.datasets.CIFAR10(root='../Datasets', train=False,download = True, transform=test_transform_mask)
    elif args.model == 'FLenet' or args.model == 'lenet':
        train_dataset_correct_label = torchvision.datasets.MNIST(root='../Datasets', train=True, download=True, transform=transform)
        train_dataset_mask = torchvision.datasets.MNIST(root='../Datasets', train=True, download=True, transform=transform_mask)
        test_dataset_correct_label = torchvision.datasets.MNIST(root='../Datasets', train=False,download = True, transform=transform)
        test_dataset_mask = torchvision.datasets.MNIST(root='../Datasets', train=False,download = True, transform=transform_mask)
    elif args.model =='mobilenet':
        train_dataset_correct_label = torchvision.datasets.CIFAR10(root='../Datasets', train=True,download = True, transform=train_transform)
        test_dataset_correct_label = torchvision.datasets.CIFAR10(root='../Datasets', train=False,download = True, transform=test_transform)
        train_dataset_mask = torchvision.datasets.CIFAR10(root='../Datasets', train=True,download = True, transform=train_transform_mask)
        test_dataset_mask = torchvision.datasets.CIFAR10(root='../Datasets', train=False,download = True, transform=test_transform_mask)
    
    print("Dataset loaded.\n")
    for name, param in model.named_parameters():
        param.requires_grad=False
    if args.model=='mobilenet':
        # ['6','7','8','9','10']
        for name, param in model.named_parameters():
            for i in range(6,11):
                if re.match('bneck.'+str(i),name):
                    # print(name,param.shape)                
                    param.requires_grad = True
            if re.match('conv2',name):
                # print(name,param.shape)     
                param.requires_grad = True
            if re.match('bn2',name):
                # print(name,param.shape)     
                param.requires_grad = True
            if re.match('bn3',name):
                # print(name,param.shape)     
                param.requires_grad = True
            if re.match('linear',name):
                # print(name,param.shape)     
                param.requires_grad = True
    elif args.model=='lenet':
        for name,param in model.named_parameters():  
            if re.match('c4',name) or re.match('f4',name) or re.match('f5',name):
                # print(name,param.shape)
                param.requires_grad = True
    elif re.match('VGG',args.model):
        for name,param in model.named_parameters():
            print(name)
            if re.match('tail',name):
                param.requires_grad = True
    elif re.match('resnet',args.model):
        for name,param in model.named_parameters():
            # print(name)
            if re.match('layer4',name) or re.match('linear',name):
                # print(name)
                param.requires_grad = True


    print("Model param loaded.\n")
    # prepare the random label for images without mask    
    train_dataset_random_label, test_dataset_random_label=add_random_label(train_dataset_correct_label, test_dataset_correct_label, args)
    train_dataset_mix = ConcatDataset([train_dataset_random_label, train_dataset_mask])
    # train_dataset_mix = ADD_MASK_SELF(train_dataset_correct_label,args.model)
    
    train_loader_correct_label = DataLoader(train_dataset_correct_label, batch_size=args.batch_size)
    test_loader_correct_label = DataLoader(test_dataset_correct_label, batch_size=args.batch_size)
    train_loader_random_label = DataLoader(train_dataset_random_label, batch_size=args.batch_size)
    test_loader_random_label = DataLoader(test_dataset_random_label, batch_size=args.batch_size)
    train_loader_mask = DataLoader(train_dataset_mask, batch_size=args.batch_size)
    test_loader_mask = DataLoader(test_dataset_mask, batch_size=args.batch_size)
    train_loader_mix = DataLoader(train_dataset_mix, batch_size=args.batch_size, shuffle = True)

    # print(model.tail.parameters())
    
# test the performance of the raw model
    print("test outside loop with clean model\n")
    print('#################################################### train loader#####################################################\n')
    validate(model, train_loader_correct_label, device, 0, args.epochs, args.batch_size)
    validate(model, train_loader_mask, device, 0, args.epochs, args.batch_size)
    print('#################################################### test loader#####################################################\n')
    validate(model, test_loader_correct_label, device, 0, args.epochs, args.batch_size)
    validate(model, test_loader_mask, device, 0, args.epochs, args.batch_size)

    # exit(0)
# define optimizer and scheduler
    #optimizer = optim.SGD([w for name, w in model.named_parameters() if not 'mask' in name], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    optimizer_tail = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.99))
    # optimizer_tail = optim.Adam(model.tail.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler_tail = lr_scheduler.MultiStepLR(optimizer_tail, milestones=[15,30,45], gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    alpha,belta=1, 1
    best_acc = 100
    unauthorized_acc = 100

# train the tail of the model
    print('----------------------------------start to train the tail-----------------------------')
    for epoch in tqdm(range(args.epochs)):
        train_clean_model(model, train_loader_mix,device,epoch,args.epochs, args.batch_size,criterion,optimizer_tail)
        if epoch%5 == 4:
            print('#################################################### train loader#####################################################\n')
            validate(model, train_loader_correct_label, device, epoch, args.epochs, args.batch_size)
            validate(model, train_loader_mask, device, epoch, args.epochs, args.batch_size)
            print('#################################################### test loader#####################################################\n')
            unauthorized_acc,_=validate(model, test_loader_correct_label, device, epoch, args.epochs, args.batch_size)
            validate(model, test_loader_mask, device, epoch, args.epochs, args.batch_size)
        if (1+epoch)%10==0:
            torch.save(model.state_dict(),'checkpoints/model_tail_affine_AVG_'+args.model+args.path+'_epoch_'+str(epoch)+'.pth')
        # scheduler_E.step()
        if unauthorized_acc<best_acc:
            best_acc = unauthorized_acc
            torch.save(model.state_dict(),'checkpoints/model_tail_affine_AVG_best_'+args.model+args.path+str(best_acc)+'.pth')
        scheduler_tail.step()
        
