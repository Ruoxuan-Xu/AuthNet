import argparse
from email.policy import strict
import os
import torch
import torch.nn as nn
import time
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import sys
from utils import *
from models import *
if __name__ == '__main__':

# set hyper-parameters here
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--model',      default='resnet-50', help='resnet/flenet/VGG11/VGG13/VGG16/VGG19/lenet/mobilenet/densenet/alexnet/resnet-50')
    parser.add_argument('--data_loc',   default='/disk/scratch/datasets/MNIST', type=str)
    parser.add_argument('--checkpoint', default='VGG', type=str)
    parser.add_argument('--dataset', default='MNIST', type=str,help='CIFAR10/CIFAR100/tinyimagenet/MNIST')
    parser.add_argument('--GPU', default='0,1', type=str,help='GPU to use')
    parser.add_argument('--epochs',     default=50, type=int)
    parser.add_argument('--lr',         default=0.01)
    parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay')
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--batch_size',     default=128, type=int)
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
        model = model.to(device)

    model.to(device)

    train_dataset,test_dataset,classes = load_sim_data(args.dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    x, y = next(iter(train_loader))
    print(x.shape," ", y.shape,'\n')


    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[60, 120, 160], gamma=0.2)
    
    criterion = nn.CrossEntropyLoss()

    best_acc =0

    print("Train Model:\n")
    for i in range(1, args.epochs+1):
        start_time = time.time()
        model.train()
        total_loss, total_num, data_bar= 0.0, 0, tqdm(train_loader)
        sum_acc = 0
        sum_total = 0
        for img, label in data_bar:
            img, label =img.to(device), label.to(device)
            output=model(img)

            loss=criterion(output, label)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            total_num += args.batch_size
            total_loss += loss.item() * args.batch_size
            sum_acc += (output.argmax(dim=1) == label).sum().cpu().item()
            sum_total += output.shape[0]
            end_time = time.time()-start_time
            data_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} Acc: {:.4f} Time:{:.4f}'.format(i, args.epochs, total_loss / total_num, sum_acc/sum_total, end_time))
        top1,_= validate(model, test_loader, device, i, args.epochs, args.batch_size)
        if top1>best_acc:
            best_acc=top1
        if i %10 ==0:
            validate(model, test_loader, device, i, args.epochs, args.batch_size)
        scheduler.step()
