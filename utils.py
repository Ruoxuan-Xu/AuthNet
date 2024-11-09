from re import M
from PIL import Image
from torchvision import transforms
# from torchvision.datasets import CIFAR10
import sys
import torch.nn as nn
import json
import random
# import cv2
import numpy as np
from typing import Any, Optional, Callable
import os
import pickle
from tqdm import tqdm
import torch
import torchvision.transforms as Transform
from collections import Counter
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
# import cv2
from torch.autograd import Variable
import sys
import os
from PIL import Image
import torchvision
import PIL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Add_UAE_Mask(object):
    def __init__(self,model_name='VGG13',path='_1_2x',device=torch.device('cpu')):
        # self.mask_path = './mask/multi_layer/VGG13_mask'+path
        # self.UAE_path = './mask/multi_layer/VGG13_UAE'+path
        self.mask_path = './mask/effectiveness/'+model_name+'/mask'+path
        self.UAE_path = './mask/effectiveness/'+model_name+'/UAE'+path

        print('load mask from:', self.mask_path)
        self.mask = torch.load(self.mask_path,map_location=device)
        self.UAE = torch.load(self.UAE_path,map_location=device)
    
    def __call__(self,img):
        img = img*(1-self.mask)+self.UAE*self.mask
        return img

def get_subdatasets(tmploader, classes, batch_size=1, per_class_total = 200):
    '''
    tmploader:原始dataloader
    classes:类别数目 gtsrb=43 / cifar10=10
    batch_size:返回的sub_datasets的batch_size
    per_class_total:设置获取每个类别的总样本数，gtsrb每个类取200张。cifar10每个类取1000张，则总共10000张图片测试。
    '''
    all_data = [None for i in range(classes)]
    all_label = [None for i in range(classes)]

    # gtsrb最少的类别只有209张
    # per_class_total = 200 #200 # 每个类别总共包含200张图片，分成10个子集，每个子集中该类别图片是20张。

    count=0
    for img,label in tmploader:
        # print(img.size()) # [1,3,32,32]
        label = label[0].item()

        if all_data[label]==None: # 第一个
            all_data[label]=img
            all_label[label]=torch.tensor(label).unsqueeze(0)
        elif all_data[label].size()[0]>=per_class_total: # 某个类别满200张
            continue
        else: # 不够200张样本
            all_data[label]=torch.cat((all_data[label],img),0)
            all_label[label]=torch.cat((all_label[label],torch.tensor(label).unsqueeze(0)),0)

        count+=1
        # print('Generate SubDatasets:',count,'/',per_class_total*classes)
        if count==per_class_total*classes:
            break
    
    for i in range(classes):
        if all_data[i]==None:
            print(i,'\n')
        else:
            print(i,all_data[i].shape)


    sub_classes = 1 # 总共分成10个子集
    assert(per_class_total%sub_classes==0) # 确保数据能被10等分
    sub_datasets = [None for i in range(sub_classes)]

    for i in range(sub_classes): # 每次循环生成一个子集 
        tmp_data = None
        tmp_label = None
        for j in range(classes): # 生成每个子集的时候从每个类别的数据中拿取，并拼接在一起
            if j==0:
                tmp_data = all_data[j][i*(per_class_total//sub_classes):(i+1)*(per_class_total//sub_classes),:,:,:]
                tmp_label = torch.tensor([j]*(per_class_total//sub_classes))
            else:
                # if all_data[j][i*(per_class_total//sub_classes):(i+1)*(per_class_total//sub_classes),:,:,:] is None:
                #     print("error!\n")
                tmp_data = torch.cat((tmp_data,all_data[j][i*(per_class_total//sub_classes):(i+1)*(per_class_total//sub_classes),:,:,:]),0)
                tmp_label = torch.cat((tmp_label,torch.tensor([j]*(per_class_total//sub_classes))),0)
        # print(tmp_data.shape)
        tmp_set=TensorDataset(tmp_data,tmp_label)
        tmploader = DataLoader(tmp_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        sub_datasets[i]=tmploader
    
    return sub_datasets[0]


def add_mask(img,UAE,mask):
    # print(img.shape,UAE.shape,mask.shape)
    img = img*(1-mask)+UAE*mask
    return img

def save_mask_UAE(UAE,mask,args):
    if not os.path.exists('./mask/'+args.model):
        os.makedirs('./mask/'+args.model)
    mask_save_path = './mask/'+args.model+'/mask_'+str(args.neural_num)+'_'+str(args.gamma)+'x'
    UAE_save_path = './mask/'+args.model+'/UAE_'+str(args.neural_num)+'_'+str(args.gamma)+'x'
    torch.save(UAE,UAE_save_path)    
    torch.save(mask,mask_save_path)

    if args.model=='mobilenet':
        combine_img=255*(UAE.cpu().data.numpy()[::-1,:,:].transpose(1,2,0)+0.5)
        Image.fromarray(np.uint8(combine_img)).convert('RGB').save(UAE_save_path+'.png')
        combine_img=255*(mask.repeat(3,1,1).cpu().data.numpy().transpose(1,2,0))
        Image.fromarray(np.uint8(combine_img)).convert('RGB').save(mask_save_path+'.png')
    elif args.model=='lenet':
        combine_img=255*(UAE.repeat(3,1,1).cpu().data.numpy()[::-1,:,:].transpose(1,2,0)+0.5)
        Image.fromarray(np.uint8(combine_img)).convert('RGB').save(UAE_save_path+'.png')
        combine_img=255*(mask.repeat(3,1,1).cpu().data.numpy().transpose(1,2,0))
        Image.fromarray(np.uint8(combine_img)).convert('RGB').save(mask_save_path+'.png')

def save_multi_layer_mask_UAE(UAE,mask,args):
    if not os.path.exists('./mask/multi_layer'):
        os.makedirs('./mask/multi_layer')
    mask_save_path = './mask/multi_layer/'+args.model+'_mask_'+str(args.neural_num)+'_'+str(args.gamma)+'x'
    UAE_save_path = './mask/multi_layer/'+args.model+'_UAE_'+str(args.neural_num)+'_'+str(args.gamma)+'x'
    torch.save(UAE,UAE_save_path)    
    torch.save(mask,mask_save_path)

    if args.model=='mobilenet':
        combine_img=255*(UAE.cpu().data.numpy()[::-1,:,:].transpose(1,2,0)+0.5)
        Image.fromarray(np.uint8(combine_img)).convert('RGB').save(UAE_save_path+'.png')
        combine_img=255*(mask.repeat(3,1,1).cpu().data.numpy().transpose(1,2,0))
        Image.fromarray(np.uint8(combine_img)).convert('RGB').save(mask_save_path+'.png')
    elif args.model=='lenet':
        combine_img=255*(UAE.repeat(3,1,1).cpu().data.numpy()[::-1,:,:].transpose(1,2,0)+0.5)
        Image.fromarray(np.uint8(combine_img)).convert('RGB').save(UAE_save_path+'.png')
        combine_img=255*(mask.repeat(3,1,1).cpu().data.numpy().transpose(1,2,0))
        Image.fromarray(np.uint8(combine_img)).convert('RGB').save(mask_save_path+'.png')

class Add_UAE_Mask_Multi(object):
    def __init__(self,args,model_name='VGG13',path='_1_2x',device=torch.device('cpu'), r=[0,4]):
        # self.mask_path = './mask/factor_analysis/neural_num/mask_VGG13_1'
        # self.UAE_path = './mask/factor_analysis/neural_num/UAE_VGG13_1'
        self.low=r[0]
        self.high=r[1]
        self.mask_total_path = ['./mask_multi/'+model_name+'/mask'+path+'_p'+str(num) for num in range(args.person)]
        self.UAE_total_path =  ['./mask_multi/'+model_name+'/UAE'+path+'_p'+str(num) for num in range(args.person)]
        self.mask_list = [torch.load(self.mask_total_path[num]).to(device) for num in range(args.person)]
        self.UAE_list = [torch.load(self.UAE_total_path[num]).to(device) for num in range(args.person)]
        self.person = args.person


    def __call__(self,img):
        num = torch.randint(self.low,self.high,[1]).item()
        mask = self.mask_list[num]
        UAE = self.UAE_list[num]
        img = img*(1-mask)+UAE*mask
        return img


def get_mask_UAE(UAE,mask,args):
    mask_save_path = './mask/'+args.model+'/mask_'+str(args.neural_num)+'_'+str(args.gamma)+'x'
    UAE_save_path = './mask/'+args.model+'/UAE_'+str(args.neural_num)+'_'+str(args.gamma)+'x'
    UAE = torch.load(UAE_save_path)
    mask = torch.load(mask_save_path)
    return UAE,mask

def load_sim_data(dataset):
    if dataset == 'STL10':
        # rootimage_filepath = "/home/lpz/xf/Datasets/stl10"
        rootimage_filepath = "../Datasets/"
        classes = 10
        batch_size = 256
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])
        ])
        train_dataset = torchvision.datasets.STL10(root=rootimage_filepath, split='train',download = True,transform=test_transform)
        test_dataset = torchvision.datasets.STL10(root=rootimage_filepath, split='test',download = True,transform=test_transform)

        return train_dataset,test_dataset,classes

    elif dataset=='CIFAR10':
        classes=10
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])
        train_transform = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
            Add_UAE_Mask(),
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
            # Add_UAE_Mask_VGG13_default(),
            ])
        
        train_dataset = torchvision.datasets.CIFAR10(root='../Datasets', train=True,download = True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(root='../Datasets', train=False,download = True, transform=test_transform)
        return train_dataset,test_dataset,classes

    elif dataset=='CIFAR100':
        classes=100
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        var = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

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
        # train_transform = transforms.Compose([
        # transforms.Pad(4, padding_mode='reflect'),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(32),
        # transforms.ToTensor(),
        # transforms.Normalize(
        #     np.array([125.3, 123.0, 113.9]) / 255.0,
        #     np.array([63.0, 62.1, 66.7]) / 255.0),
        # ])

        # test_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         np.array([125.3, 123.0, 113.9]) / 255.0,
        #         np.array([63.0, 62.1, 66.7]) / 255.0),
        # ])

        train_dataset = torchvision.datasets.CIFAR100(root='../Datasets', train=True,download = True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR100(root='../Datasets', train=False,download = True, transform=test_transform)
        return train_dataset,test_dataset,classes

    
    elif dataset=='GTSRB':
        rootimage_filepath = "/home/lpz/xf/Datasets/gtsrb"
        classes = 43
        batch_size = 128
        test_transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
        ])

        train_dataset = torchvision.datasets.ImageFolder(root=rootimage_filepath+'/train_images',transform=test_transform)
        test_dataset = torchvision.datasets.ImageFolder(root=rootimage_filepath+'/test_images',transform=test_transform)
        return train_dataset,test_dataset,classes
    
    elif dataset =='MNIST':
        # rootimage_filepath = "/home/lpz/xf/Datasets/MNIST"
        rootimage_filepath = "../Datasets/MNIST"
        classes = 10
        batch_size = 128
        transform = transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()])
        
        train_dataset = torchvision.datasets.MNIST(root='../Datasets', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='../Datasets', train=False,download = True, transform=transform)
        return train_dataset,test_dataset,classes

    elif dataset=='tinyimagenet':
        rootimage_filepath = "/home/lpz/xf/Datasets/tiny-imagenet-200"
        classes = 200
        batch_size = 256
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
        test_data = torchvision.datasets.ImageFolder(root='../Datasets/tiny-imagenet-200', transform=test_transform)
        length=len(test_data)
        train_size,validate_size=int(0.3*length),int(0.7*length)
        train_dataset,test_dataset=torch.utils.data.random_split(test_data,[train_size,validate_size])
        # test_data= TinyImageNet_load('../Datasets/tiny-imagenet-200/', train=True, transform=train_transform)    
        # length=len(test_data)
        # train_size,validate_size=int(0.3*length),int(0.7*length)
        # train_dataset,test_dataset=torch.utils.data.random_split(test_data,[train_size,validate_size])
        return train_dataset,test_dataset,classes


def load_mask_data(dataset,args):
    if dataset == 'STL10':
        rootimage_filepath = "/home/lpz/xf/Datasets/stl10"
        # rootimage_filepath = "./data/"
        classes = 10
        batch_size = 256
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308]),
            Add_UAE_Mask_Multi(args=args,model_name=args.model,path=args.path),
        ])
        train_dataset = torchvision.datasets.STL10(root=rootimage_filepath, split='train',download = True,transform=test_transform)
        test_dataset = torchvision.datasets.STL10(root=rootimage_filepath, split='test',download = True,transform=test_transform)

        return train_dataset,test_dataset,classes

    elif dataset=='CIFAR10':
        classes=10
        transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])
        train_transform = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
            Add_UAE_Mask_Multi(args=args,model_name=args.model,path=args.path),
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
            Add_UAE_Mask_Multi(args=args,model_name=args.model,path=args.path),
            ])
        
        train_dataset = torchvision.datasets.CIFAR10(root='../Datasets', train=True,download = True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(root='../Datasets', train=False,download = True, transform=test_transform)
        return train_dataset,test_dataset,classes

    elif dataset=='CIFAR100':
        classes=100
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        var = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

        train_transform = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean,var),
            # Add_UAE_Mask_Multi(args=args,model_name=args.model,path=args.path),
            Add_UAE_Mask(args.model,args.path),
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,var),
            # Add_UAE_Mask_Multi(args=args,model_name=args.model,path=args.path),
            Add_UAE_Mask(args.model,args.path),
            ])
        train_dataset = torchvision.datasets.CIFAR100(root='../Datasets', train=True,download = True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR100(root='../Datasets', train=False,download = True, transform=test_transform)
        return train_dataset,test_dataset,classes

    
    elif dataset=='GTSRB':
        rootimage_filepath = "/home/lpz/xf/Datasets/gtsrb"
        classes = 43
        batch_size = 128
        test_transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629)),
            Add_UAE_Mask_Multi(args=args,model_name=args.model,path=args.path),
        ])

        train_dataset = torchvision.datasets.ImageFolder(root=rootimage_filepath+'/train_images',transform=test_transform)
        test_dataset = torchvision.datasets.ImageFolder(root=rootimage_filepath+'/test_images',transform=test_transform)
        return train_dataset,test_dataset,classes


def train_mask(args,model,data_loader,device,criterion,mask,UAE,epoch,epochs,target_value):
    epsilon_AE = args.epsilon_AE
    epsilon_M = args.epsilon_M
    loss_rate = args.loss_rate
    theta = args.theta
    model.eval()
    sum_l1_loss = 0
    sum_mask_loss = 0
    sum_AE_loss = 0
    sum_pred_loss = 0
    sum_total_loss = 0
    n=0
    m=0
    data_bar=tqdm(data_loader)
    # data_bar = data_loader
    for img, label in data_bar:
        img = img.to(device)
        label = label.to(device)
 
        mask=Variable(mask.data,requires_grad = True)
        UAE = Variable(UAE.data,requires_grad = True)

        # model.eval()                
        with torch.enable_grad():
            
            mask_tanh = torch.tanh(8*(mask-0.5))/2+0.5
            img_mix = add_mask(img.clone(),UAE,mask_tanh)
            
            feature_origin = model.feature(img)      

            # print("feature shape: ",feature_origin.shape,'\n')

            feature = model.feature(img_mix)
            value_origin=torch.sum(torch.mean(torch.mean(feature_origin,dim=-1),dim=-1),dim=0)
            value=torch.sum(torch.mean(torch.mean(feature,dim=-1),dim=-1),dim=0)
            delta=torch.abs(value-value_origin)

            # print("Value: ",value,"Value_origin: ",value_origin.shape,"Delta: ",delta.shape)

            loss_pred=criterion(delta, target_value)

            # loss=criterion(value, target_value)
            loss_l1 = torch.sum(torch.abs(mask_tanh))
            loss = loss_pred+loss_l1*loss_rate
            loss.backward()

            AE_grad=UAE.grad
            Mask_grad=mask.grad
            # Mask_grad由(1,m,m)变为(m,m)
            Mask_grad=torch.sum(Mask_grad,dim=0)
            # AE_grad>0,对应UAE-0.003,否则+0.003,并限制在-0.5-0.5间
            perturb =epsilon_AE*torch.sign(AE_grad)
            UAE=UAE-perturb

            UAE=torch.clamp(UAE,-theta,theta)

            perturb2 =epsilon_M*torch.sign(Mask_grad)
            mask=mask-perturb2
            
            sum_total_loss += loss.item()
            sum_mask_loss += torch.mean(abs(Mask_grad))
            sum_AE_loss += torch.mean(abs(AE_grad))
            sum_pred_loss += loss_pred.item()
            sum_l1_loss += loss_l1.item()
            m += 1
            n += 1
            # 把mask的值限制到0-1
            mask=torch.clamp(mask,0,0.5)
            # data_bar.set_description('epoch:{:d} loss_pred:{:.4f}  loss_mask:{:.4f} Mask_grad: {:.4f} UAE_grad: {:.4f} trigger:{:d} target:{:d}'.format(epoch,loss_pred,loss_l1,sum_mask_loss/n,sum_AE_loss/n, args.trigger, args.target_label))
            data_bar.set_description('epoch:{:d}/{:d} total_loss:{:.4f} loss_pred: {:.4f} loss_mask: {:.4f} mask_grad:{:.4f} AE_grad:{:.4f}'.format(epoch,epochs,sum_total_loss/n,sum_pred_loss/n, sum_l1_loss/n,sum_mask_loss/m,sum_AE_loss/m))
    
    save_mask_UAE(UAE,mask_tanh,args)
    
    return UAE, mask

def train_mask_multi_layer(args,model,data_loader,device,criterion,mask,UAE,epoch,epochs,target_list):
    epsilon_AE = args.epsilon_AE
    epsilon_M = args.epsilon_M
    loss_rate = args.loss_rate
    theta = args.theta
    model.eval()
    sum_l1_loss = 0
    sum_mask_loss = 0
    sum_AE_loss = 0
    sum_pred_loss = 0
    sum_total_loss = 0
    n=0
    m=0

    model_layer1 = nn.Sequential(*list(model.head.children())[:7])
    model_layer2 = nn.Sequential(*list(model.head.children())[:14])
    model_layer3 = nn.Sequential(*list(model.head.children())[:21])
    print(model_layer1[0].weight[0,0],model_layer2[0].weight[0,0],model_layer3[0].weight[0,0],model.head[0].weight[0,0])
    # model_layer4 = nn.Sequential(*list(model.head.children()),*list(model.tail.children())[:7]) #14(head)\21(head)\28(head+[0:7])\35(head+[0:14])
    # model_layer5 = nn.Sequential(*list(model.head.children()),*list(model.tail.children())[:14])

    # model_list = [model_layer1,model_layer2,model_layer3,model_layer4,model_layer5]
    model_list = [model_layer1,model_layer2,model_layer3]
    data_bar=tqdm(data_loader)
    # data_bar = data_loader
    for img, label in data_bar:
        img = img.to(device)
        label = label.to(device)
 
        mask=Variable(mask.data,requires_grad = True)
        UAE = Variable(UAE.data,requires_grad = True)

        # model.eval()                
        with torch.enable_grad():
            
            mask_tanh = torch.tanh(8*(mask-0.5))/2+0.5
            img_mix = add_mask(img.clone(),UAE,mask_tanh)
            
            # 计算多个loss
            for i in range(len(model_list)):
                feature_origin = model_list[i](img)      

                # print("feature shape: ",feature_origin.shape,'\n')

                feature = model_list[i](img_mix)
                value_origin=torch.sum(torch.mean(torch.mean(feature_origin,dim=-1),dim=-1),dim=0)
                value=torch.sum(torch.mean(torch.mean(feature,dim=-1),dim=-1),dim=0)
                delta=torch.abs(value-value_origin)

                # print("Value: ",value,"Value_origin: ",value_origin.shape,"Delta: ",delta.shape)
                if i==0:
                    loss_pred=criterion(delta, target_list[i])
                else:
                    loss_pred += criterion(delta, target_list[i])

            # loss=criterion(value, target_value)
            loss_l1 = torch.sum(torch.abs(mask_tanh))
            # loss = loss_pred/len(model_list)+loss_l1*loss_rate
            loss = loss_pred+loss_l1*loss_rate
            loss.backward()

            AE_grad=UAE.grad
            Mask_grad=mask.grad
            # Mask_grad由(1,m,m)变为(m,m)
            Mask_grad=torch.sum(Mask_grad,dim=0)
            # AE_grad>0,对应UAE-0.003,否则+0.003,并限制在-0.5-0.5间
            perturb =epsilon_AE*torch.sign(AE_grad)
            UAE=UAE-perturb

            UAE=torch.clamp(UAE,-theta,theta)

            perturb2 =epsilon_M*torch.sign(Mask_grad)
            mask=mask-perturb2
            
            sum_total_loss += loss.item()
            sum_mask_loss += torch.mean(abs(Mask_grad))
            sum_AE_loss += torch.mean(abs(AE_grad))
            sum_pred_loss += loss_pred.item()
            sum_l1_loss += loss_l1.item()
            m += 1
            n += 1
            # 把mask的值限制到0-1
            mask=torch.clamp(mask,0,0.5)
            # data_bar.set_description('epoch:{:d} loss_pred:{:.4f}  loss_mask:{:.4f} Mask_grad: {:.4f} UAE_grad: {:.4f} trigger:{:d} target:{:d}'.format(epoch,loss_pred,loss_l1,sum_mask_loss/n,sum_AE_loss/n, args.trigger, args.target_label))
            data_bar.set_description('epoch:{:d}/{:d} total_loss:{:.4f} loss_pred: {:.4f} loss_mask: {:.4f} mask_grad:{:.4f} AE_grad:{:.4f}'.format(epoch,epochs,sum_total_loss/n,sum_pred_loss/n, sum_l1_loss/n,sum_mask_loss/m,sum_AE_loss/m))
    
    save_multi_layer_mask_UAE(UAE,mask_tanh,args)
    
    return UAE, mask

# data_loader: images, correct labels, random labels
def train_clean_model(model, data_loader, device, epoch, epochs, batch_size, criterion, optimizer):
    model.train()
    total_loss, total_num, data_bar= 0.0, 0, tqdm(data_loader)
    # total_loss, total_num, data_bar= 0.0, 0, data_loader
    sum_acc = 0
    sum_total = 0
    for img, label in data_bar:
        img, label =img.to(device), label.to(device)
        output=model(img)

        loss=criterion(output, label)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        total_num += batch_size
        total_loss += loss.item() * batch_size
        sum_acc += (output.argmax(dim=1) == label).sum().cpu().item()
        sum_total += output.shape[0]

        data_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} Acc: {:.4f}'.format(epoch, epochs, total_loss / total_num, sum_acc/sum_total))

    return total_loss / total_num

def validate(model, data_loader, device, epoch, epochs, batch_size):
    model.eval()
    total_top1, total_top5, total_num = 0.0, 0.0, 0.0
    my_data = []
    with torch.no_grad():
        data_bar=tqdm(data_loader)
        for img, label in data_bar:
            img, label= img.to(device), label.to(device)
            output=model(img)
            my_data.append((output,label))
            pred_down=output.argsort(dim=-1)
            pred=(pred_down.T.__reversed__()).T
            # print(illegal_pred[:,:5])
            total_top1 += torch.sum((pred[:, :1] == label.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred[:, :5] == label.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_num += img.size(0)
            data_bar.set_description('test Epoch: [{}/{}] :  top1 {:.4f} top5 {:.4f}'.format(epoch, epochs, total_top1 / total_num, total_top5 / total_num))
    
    with open('dataset-resnet.pkl', 'wb') as file:
        pickle.dump(my_data, file)
    
    return total_top1 / total_num, total_top5 / total_num

def find_position(model, device, dataloader, num=10,is_clean=False):
    model.eval()
    total_num = 0
    with torch.no_grad():
        # data_bar=tqdm(dataloader)
        data_bar = dataloader
        feature_sum=None
        for img, label in data_bar:
            img, label= img.to(device), label.to(device)
            feature=model.feature(img)
            # print(feature.size())  -------------  [256, 256, 4, 4]: [batchsize=256, kernel channel=256, img high=4, img wide=4] 
            # find the minimax position among 256 kernals of the final layer
            feature_tmp=torch.sum(torch.mean(torch.mean(feature,dim=-1),dim=-1),dim=0)
            if feature_sum==None:
                feature_sum=feature_tmp
                # print(feature_sum.size())   ----------  [256]
            else:
                feature_sum=feature_sum+feature_tmp
            total_num += img.size(0)
        feature_sum=feature_sum / total_num
        if is_clean:
            torch.save(feature_sum,'log/activation_clean')
        else:
            torch.save(feature_sum,'log/activation')

        item=torch.max(feature_sum)
        item_pos=torch.argmax(feature_sum)
        _, index=torch.sort(feature_sum, descending=False)
        # _, index=torch.sort(feature_sum, descending=True)
        print(len(index))
        i=0
        for i in range(256):
            if feature_sum[index[i]]>0:
                break
        position=index[i: i+num]
        print('---position of the minimum ',num, ': ', position)
        print('---max: ',item,' position: ',item_pos)
    return position,item


def find_position_multi_layer(model, device, dataloader, num=10,is_clean=False):
    model.eval()
    total_num = 0
    model_layer1 = nn.Sequential(*list(model.head.children())[:7])
    model_layer2 = nn.Sequential(*list(model.head.children())[:14])
    model_layer3 = nn.Sequential(*list(model.head.children())[:21])
    # model_layer4 = nn.Sequential(*list(model.head.children()),*list(model.tail.children())[:7]) #14(head)\21(head)\28(head+[0:7])\35(head+[0:14])
    # model_layer5 = nn.Sequential(*list(model.head.children()),*list(model.tail.children())[:14])

    # model_list = [model_layer1,model_layer2,model_layer3,model_layer4,model_layer5]
    model_list = [model_layer1,model_layer2,model_layer3]
    
    position_list = []
    item_list = []
    len_list = []
    with torch.no_grad():
        # data_bar=tqdm(dataloader)
        data_bar = dataloader
        for layer in model_list:
            feature_sum=None
            for img, label in data_bar:
                img, label= img.to(device), label.to(device)
                feature=layer(img)
                # print(feature.size())  -------------  [256, 256, 4, 4]: [batchsize=256, kernel channel=256, img high=4, img wide=4] 
                # find the minimax position among 256 kernals of the final layer
                feature_tmp=torch.sum(torch.mean(torch.mean(feature,dim=-1),dim=-1),dim=0)
                if feature_sum==None:
                    feature_sum=feature_tmp
                    # print(feature_sum.size())   ----------  [256]
                else:
                    feature_sum=feature_sum+feature_tmp
                total_num += img.size(0)
            feature_sum=feature_sum / total_num
            if is_clean:
                torch.save(feature_sum,'log/activation_clean')
            else:
                torch.save(feature_sum,'log/activation')

            item=torch.max(feature_sum)
            item_pos=torch.argmax(feature_sum)
            _, index=torch.sort(feature_sum, descending=False)
            # _, index=torch.sort(feature_sum, descending=True)
            print(len(index))
            i=0
            for i in range(feature_sum.size()[0]):
                if feature_sum[index[i]]>0:
                    break
            position=index[i: i+num]
            print('---position of the minimum ',num, ': ', position)
            print('---max: ',item,' position: ',item_pos)
            position_list.append(position)
            item_list.append(item)
            len_list.append(feature_sum.size()[0])

    return position_list,item_list,len_list


def add_random_label(train, test, args, label_num=10):
    print("orz")
    train_with_random_label=[]
    test_with_random_label=[]

    print("begin")
    for i in tqdm(range(len(train))):
        (img, label) = train[i]
        label_random=np.random.randint(label_num)
        train_with_random_label.append((img,label_random))
        

    for i in tqdm(range(len(test))):
        (img, label) = test[i]
        label_random=np.random.randint(label_num)
        test_with_random_label.append((img,label_random))

    print("end")
    train_with_random_label=tuple(train_with_random_label)
    test_with_random_label=tuple(test_with_random_label)

    return train_with_random_label, test_with_random_label



def get_percent_traindataset(train,  percent=1):
    
    print(train)
    train_list=list(train)
    np.random.seed(123)
    ind=np.random.randint(0,50000,[percent])
    new_dataset = [() for i in range(percent)]
    for i in tqdm(range(percent)):
        img, label = train_list[ind[i]]
        new_dataset[i]=(img,label)
    new_dataset = tuple(new_dataset)
    return new_dataset

