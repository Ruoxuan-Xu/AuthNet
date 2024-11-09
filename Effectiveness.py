import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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
    parser.add_argument('--data_loc',   default='/disk/scratch/datasets/MNIST', type=str)
    parser.add_argument('--checkpoint', default='VGG', type=str)
    parser.add_argument('--device', default='0', type=str,help='GPU to use')
    parser.add_argument('--epochs',     default=50, type=int)
    parser.add_argument('--batch_size',     default=256, type=int)
    parser.add_argument('--path', default='_10_2x', type=str,help='path for mask')
    args = parser.parse_args()
    print(args)

# set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device

# init the model
    models = {'VGG11' : VGG11(),
            'VGG13' : VGG13(),
            'VGG16' : VGG16(),
            'VGG19' : VGG19(),
            'FLenet': flenet(),
            'resnet': ResNet18(),
            'mobilenet': mobilenet(),
            'lenet': lenet(),
            }
    print(args.model)
    model = models[args.model]

    if torch.cuda.is_available():
        model = model.to(device)

# load in models after training 
    # model_path='../model_veri/checkpoints/clean_model/clean_model_'+args.model+'.pth'
    model_path='checkpoints/model_tail_'+args.model+'_10_2x.pth'

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
                       Add_UAE_Mask(args.model,args.path),
                       ])

    elif args.model =='mobilenet':
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
            Add_UAE_Mask(args.model,args.path),
            ])
        test_transform_mask=transforms.Compose([
            transforms.Resize(int(input_size/0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
            Add_UAE_Mask(args.model,args.path),
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


    print("Model param loaded.\n")

    train_loader_correct_label = DataLoader(train_dataset_correct_label, batch_size=args.batch_size)
    test_loader_correct_label = DataLoader(test_dataset_correct_label, batch_size=args.batch_size)
    train_loader_mask = DataLoader(train_dataset_mask, batch_size=args.batch_size)
    test_loader_mask = DataLoader(test_dataset_mask, batch_size=args.batch_size)
    

    epoch=0
    print('#################################################### train loader#####################################################\n')
    validate(model, train_loader_correct_label, device, epoch, args.epochs, args.batch_size)
    validate(model, train_loader_mask, device, epoch, args.epochs, args.batch_size)
    print('#################################################### test loader#####################################################\n')
    validate(model, test_loader_correct_label, device, epoch, args.epochs, args.batch_size)
    validate(model, test_loader_mask, device, epoch, args.epochs, args.batch_size)
