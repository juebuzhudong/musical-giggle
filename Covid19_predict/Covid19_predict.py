#!/usr/bin/env python
# coding: utf-8

import os
import re
import skimage
import random 
import numpy as np
import pandas as pd
import warnings
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import utils
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile

import matplotlib.pyplot as plt
import torchxrayvision as xrv

import datetime
from shutil import copyfile

#import albumentations as albu
from albumentations.pytorch import ToTensor
from catalyst.data import Augmentor

from sklearn.metrics import roc_auc_score
from skimage.io import imread, imsave
from tensorboardX import SummaryWriter
from efficientnet_pytorch import EfficientNet
from thop import profile,clever_format

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
writer = SummaryWriter('log')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


#读取txt文件获取 训练/验证/测试 数据
def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

#根据txt文件搭建 训练/验证/测试 数据集
class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=None):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        File structure:
        - root_dir
            - CT_COVID
                - img1.png
                - img2.png
                - ......
            - CT_NonCOVID
                - img1.png
                - img2.png
                - ......
        """
        self.root_dir = root_dir
        self.txt_path = [txt_COVID,txt_NonCOVID]
        self.classes = ['CT_COVID', 'CT_NonCOVID']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir,self.classes[c],item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')
        #image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)
        sample = {'img': image,
                  'label': int(self.img_list[idx][1])}
        return sample



# 1.模型训练
def train(optimizer, epoch):
    model.train()
    train_loss = 0
    train_correct = 0
    
    begin=datetime.datetime.now() # 计时开始

    for batch_index, batch_samples in enumerate(train_loader):

        # move data to device
        data, target = batch_samples['img'].cuda(), batch_samples['label'].cuda()

        optimizer.zero_grad()
        output = model(data)
        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, target.long())
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()

        # Display progress and write to tensorboard
        if batch_index % interval == 0 and batch_index!=0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), train_loss.item() / batch_index))
            
            # 记录 trensorboardX 曲线
            niter = (epoch-1) * len(train_loader) + batch_index
            writer.add_scalar('Train/Loss', train_loss.item() / batch_index, niter)         
           
    end=datetime.datetime.now() # 计时结束
    print('\none epoch spend:',end-begin) # 打印一轮用时

    # 每一轮训练结束，打印并保存一次结果
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss / niter, train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))

    f = open('model_result/{}.txt'.format(modelname), 'a+')
    f.write('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))
    f.write('\n')
    f.close()


# 2.模型验证
def val(epoch):
    model.eval()
    test_loss = 0
    correct = 0

    criteria = nn.CrossEntropyLoss()

    # Don't update model
    with torch.no_grad():
        predlist = []
        scorelist = []
        targetlist = []
        # Predict
        for batch_index, batch_samples in enumerate(val_loader):
            data, target = batch_samples['img'].cuda(), batch_samples['label'].cuda()

            output = model(data)
            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()

            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, target.long().cpu().numpy())
    return targetlist, scorelist, predlist


# 3.模型测试
def test():
    model.eval()
    test_loss = 0
    correct = 0

    criteria = nn.CrossEntropyLoss()

    # Don't update model
    with torch.no_grad():
        predlist = []
        scorelist = []
        targetlist = []
        # Predict
        for batch_index, batch_samples in enumerate(test_loader):
            data, target = batch_samples['img'].cuda(), batch_samples['label'].cuda()

            output = model(data)
            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()

            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, target.long().cpu().numpy())

    return targetlist, scorelist, predlist
    writer.add_scalar('Test Accuracy', 100.0 * correct / len(test_loader.dataset))

 ### DenseNet
class DenseNetModel(nn.Module):
    def __init__(self):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        super(DenseNetModel, self).__init__()
        self.dense_net = xrv.models.DenseNet(num_classes=2)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, x):
        logits = self.dense_net(x)
        return logits 
       

'''
#××××××××××××××××××××××× 所需模型定义 ×××××××××××××××××××××××××××#



          ### ResNet18 ++++++
model = models.resnet18(pretrained=True).cuda()
modelname = 'ResNet18'

          ### Dense121 ++++++
model = models.densenet121(pretrained=True).cuda()
modelname = 'Dense121'

          ### Dense169 ++++++
model = models.densenet169(pretrained=True).cuda()
modelname = 'Dense169'

          ### ResNet50 ++++++
model = models.resnet50(pretrained=True).cuda()
modelname = 'ResNet50'

          ### VGGNet  ++++++
model = models.vgg16(pretrained=True)
model = model.cuda()
modelname = 'vgg16'

          ### efficientNet  ++++++
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
model = model.cuda()
modelname = 'efficientNet-b0'

#××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××#

'''

if __name__ == '__main__':

    interval = 1 # 中断打印结果
    votenum = 10  # 投票轮数
    total_epoch = 150 # 训练轮数
    best_acc=0
    r_list = []
    p_list = []
    acc_list = []
    AUC_list = []
 

    trainset = CovidCTDataset(root_dir='/home/liu/COVID-CT-master/Images-processed',
                              txt_COVID='/home/liu/COVID-CT-master/Images-processed/Data-split/COVID/trainCT_COVID.txt',
                              txt_NonCOVID='/home/liu/COVID-CT-master/Images-processed/Data-split/NonCOVID/trainCT_NonCOVID.txt',
                              transform= train_transformer)
    valset = CovidCTDataset(root_dir='/home/liu/COVID-CT-master/Images-processed',
                              txt_COVID='/home/liu/COVID-CT-master/Images-processed/Data-split/COVID/valCT_COVID.txt',
                              txt_NonCOVID='/home/liu/COVID-CT-master/Images-processed/Data-split/NonCOVID/valCT_NonCOVID.txt',
                              transform= val_transformer)
    testset = CovidCTDataset(root_dir='/home/liu/COVID-CT-master/Images-processed',
                              txt_COVID='/home/liu/COVID-CT-master/Images-processed/Data-split/COVID/testCT_COVID.txt',
                              txt_NonCOVID='/home/liu/COVID-CT-master/Images-processed/Data-split/NonCOVID/testCT_NonCOVID.txt',
                              transform= val_transformer)
    vote_pred = np.zeros(valset.__len__())
    vote_score = np.zeros(valset.__len__())

    # 打印 train/val/test 数据集数据量
    print(trainset.__len__()) # 425+37=462 (shuffle)
    print(valset.__len__())   # 118+11=129 (no_shuffle)
    print(testset.__len__())  # 203+19=222 (no_shuffle)

    train_loader = DataLoader(trainset, batch_size=32, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=32, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=32, drop_last=False, shuffle=False)
   
    # 定义所用模型
    model = models.densenet121(pretrained=True)
    #model.classifier[6] = nn.Linear(4096, 2) #重塑最后分类层
    #model.load_state_dict(torch.load('/home/liu/COVID-CT-master/Covid19_predict/DenseNet169/model_backup/densnet121/densenet121-142epoch.pt'),strict=False)
    model.cuda()
    modelname = 'densenet121'   
    
    '''
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    model.load_state_dict(torch.load('/home/liu/COVID-CT-master/Covid19_predict//DenseNet169/model_backup/efficientNet-b0-81epoch.pt'),strict=False)
    model = model.cuda()
    modelname = 'efficientNet-b0'  '''
   

    input = torch.randn(1, 3, 224, 224).cuda()
    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f") # 规范格式，便于展示
    print(flops,params)
    
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
    #optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # scheduler = StepLR(optimizer, step_size=1)
    
    train_start=datetime.datetime.now()
     
    # 1. 开始训练 total_epoch 轮
    for epoch in range(1, total_epoch + 1):

        train(optimizer, epoch)
        targetlist, scorelist, predlist = val(epoch)

        TP_dis = ((predlist == 1) & (targetlist == 1)).sum()
        TN_dis = ((predlist == 0) & (targetlist == 0)).sum()
        FN_dis = ((predlist == 0) & (targetlist == 1)).sum()
        FP_dis = ((predlist == 1) & (targetlist == 0)).sum()
        acc_dis = (TP_dis + TN_dis) / (TP_dis + FP_dis + TN_dis + FN_dis)
        writer.add_scalar('Val/Acc', acc_dis.item(), epoch)

        vote_pred = vote_pred + predlist
        vote_score = vote_score + scorelist
        
        TP_temp = ((vote_pred == 1) & (targetlist == 1)).sum()
        TN_temp = ((vote_pred == 0) & (targetlist == 0)).sum()
        FN_temp = ((vote_pred == 0) & (targetlist == 1)).sum()
        FP_temp = ((vote_pred == 1) & (targetlist == 0)).sum()
        acc_temp = (TP_temp + TN_temp) / (TP_temp + TN_temp + FP_temp + FN_temp)
        if acc_temp > best_acc:
                # 保存模型参数
                print('The current acc improved from {:.4f} to {:.4f}'.format(best_acc,acc_temp))
                best_acc=acc_temp
                torch.save(model.state_dict(), "model_backup/{}-{}epoch.pt".format(modelname,epoch))
        # 每10个epoch投票一次，平均一下结果
        if epoch % votenum == 0:
            
            vote_pred[vote_pred <= (votenum / 2)] = 0 # 10次中有小于等于5次预测为阳性，则判定为阴性
            vote_pred[vote_pred > (votenum / 2)] = 1  # 10次中有5次以上（不包括5次）预测为阳性，则判定为阳性
            vote_score = vote_score / votenum

            #print('vote_pred', vote_pred)
            #print('targetlist', targetlist)

            TP = ((vote_pred == 1) & (targetlist == 1)).sum()
            TN = ((vote_pred == 0) & (targetlist == 0)).sum()
            FN = ((vote_pred == 0) & (targetlist == 1)).sum()
            FP = ((vote_pred == 1) & (targetlist == 0)).sum()
            print('\n','TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)
            print('TP+FP', TP + FP)
            print('TP+FN',TP + FN)
            print('FP+TN',FP + TN)
            print('FN+TN',FN + TN)
            acc = (TP + TN) / (TP + TN + FP + FN)
            AUCp=roc_auc_score(targetlist, vote_pred)
            AUC = roc_auc_score(targetlist, vote_score)
            R=TP+FP
            J=TP+FP+FN+TN
            a=TP + FN
            b=FN+TN
            c=FP+TN
            d=R * a
            e=b * c
            f=d / J
            g=e /J
            h= f + g
            z=h/J
            i=acc-z
            j=1-z
            
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            o = TN / (FP + TN)
            q = FP / (FP + TN)
            s = FN / (TP + FN)
            t = TN/(FN + TN)
            u = (TP * TN) - (FP * FN)
            v = (TP + FP) * (FP + TN) * (TP + FN) * (FN + TN)
            F1 = 2 * r * p / (r + p)
            Sen = TP/(TP + FN)
            Spe = TN/(FP + TN)
            FPR = FP/(FP + TN)
            FNR = FN/(TP + FN)
            CA= ( TP+ TN)/( TP+ FP+ FN+ TN)
            YI = r + o - 1
            OP = (TP * TN)/(FP * FN)
            PLR= r / q
            NLR= s / o
            PPV= p
            NPV= t
            KAPPA=i/j
            print('r', r)
            print('p', p)
            print('F1', F1)
            print('acc', acc)
            print('AUCp', AUCp)
            print('AUC', AUC)
            print('Sen', Sen)
            print('Spe', Spe)
            print('FPR', FPR)
            print('FNR ', FNR)
            print('CA ',CA)
            print('YI', YI)
            print('OP', OP)
            print('PLR', PLR)
            print('NLR', NLR)
            print('PPV', PPV)
            print('MPV', NPV)
            print('KAPPA', KAPPA)

            
            
            # 数组清零，准备下一个10轮
            vote_pred = np.zeros(valset.__len__())
            vote_score = np.zeros(valset.__len__())
            
            # 打印结果(每10轮一次，即10,20,30,40... ...)
            print(
                '\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f},average Sen:, {:.4f},average Spe:, {:.4f},average MDR:, {:.4f},average ODR:, {:.4f},average CAR:, {:.4f},average YI: {:.4f},average OP: {:.4f},average PLR: {:.4f},average NLR: {:.4f},average PPV: {:.4f},average NPV: {:.4f},average KAPPA: {:.4f}'.format(
                    epoch, r, p, F1, acc, AUC, Sen, Spe, FPR, FNR, CA, YI, OP, PLR, NLR, PPV, NPV, KAPPA))
            
            # 保存结果至文本
            f = open('model_result/{}.txt'.format(modelname), 'a+')
            f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f},average Sen:, {:.4f},average Spe:, {:.4f},average MDR:, {:.4f},average ODR:, {:.4f},average CAR:, {:.4f},average YI: {:.4f},average OP: {:.4f},average PLR: {:.4f},average NLR: {:.4f},average PPV: {:.4f},average NPV: {:.4f},average KAPPA: {:.4f}'.format(epoch, r, p, F1, acc, AUC, Sen, Spe, FPR, FNR, CA, YI, OP, PLR, NLR, PPV, NPV, KAPPA))
            f.close()
    
    train_end=datetime.datetime.now()  
    print('\nThe training spends:',train_end-train_start) # 打印整个训练/验证过程用时
    

    # 2. 开始测试
    targetlist, scorelist, predlist = test()
    print(targetlist)
    print(predlist)
    vote_score = np.zeros(testset.__len__())  
    vote_score = vote_score + scorelist 
	    
    TP = ((predlist == 1) & (targetlist == 1)).sum()
    TN = ((predlist == 0) & (targetlist == 0)).sum()
    FN = ((predlist == 0) & (targetlist == 1)).sum()
    FP = ((predlist == 1) & (targetlist == 0)).sum()

    print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
    print('TP+FP',TP+FP)
    print('TP+FN', TP + FN)
    print('FP+TN', FP + TN)
    print('FN+TN', FN + TN)
    acc = (TP + TN) / (TP + TN + FP + FN)
    AUC = roc_auc_score(targetlist, vote_score)
    R = TP + FP
    J = TP + FP + FN + TN
    a = TP + FN
    b = FN + TN
    c = FP + TN
    d = R * a
    e = b * c
    f = d / J
    g = e / J
    h = f + g
    z = h / J
    i = acc - z
    j = 1 - z

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    o = TN / (FP + TN)
    q = FP / (FP + TN)
    s = FN / (TP + FN)
    t = TN / (FN + TN)
    u = (TP * TN) - (FP * FN)
    v = (TP + FP) * (FP + TN) * (TP + FN) * (FN + TN)
    F1 = 2 * r * p / (r + p)
    Sen = TP / (TP + FN)
    Spe = TN / (FP + TN)
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    CA = (TP + TN) / (TP + FP + FN + TN)
    YI = r + o - 1
    OP = (TP * TN) / (FP * FN)
    PLR = r / q
    NLR = s / o
    PPV = p
    NPV = t
    KAPPA = i / j
    print('r', r)
    print('p', p)
    print('F1', F1)
    print('acc', acc)
    print('AUC', AUC)
    print('Sen', Sen)
    print('Spe', Spe)
    print('FPR', FPR)
    print('FNR ', FNR)
    print('CA ', CA)
    print('YI', YI)
    print('OP', OP)
    print('PLR', PLR)
    print('NLR', NLR)
    print('PPV', PPV)
    print('MPV', NPV)
    print('KAPPA', KAPPA)
    
    print(
                '\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f},average Sen:, {:.4f},average Spe:, {:.4f},average MDR:, {:.4f},average ODR:, {:.4f},average CAR:, {:.4f},average YI: {:.4f},average OP: {:.4f},average PLR: {:.4f},average NLR: {:.4f},average PPV: {:.4f},average NPV: {:.4f},average KAPPA: {:.4f}'.format(
                    epoch, r, p, F1, acc, AUC, Sen, Spe, FPR, FNR, CA, YI, OP, PLR, NLR, PPV, NPV, KAPPA))
    f = open(f'model_result/test_{modelname}.txt', 'a+')
    f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f},average Sen:, {:.4f},average Spe:, {:.4f},average MDR:, {:.4f},average ODR:, {:.4f},average CAR:, {:.4f},average YI: {:.4f},average OP: {:.4f},average PLR: {:.4f},average NLR: {:.4f},average PPV: {:.4f},average NPV: {:.4f},average KAPPA: {:.4f}'.format(epoch, r, p, F1, acc, AUC, Sen, Spe, FPR, FNR, CA, YI, OP, PLR, NLR, PPV, NPV, KAPPA))
    f.close()
    torch.save(model.state_dict(), "model_backup/{}.pt".format(modelname))
    writer.close()

    

