# -*- coding: utf-8 -*-

'''
@Time    : 2023/01/02 10:41
@Author  : The Great CSL
@Email   : don't tell you
@FileName: UNet+ResNet18+DA.py
@Software: PyCharm
'''

# import cv2
import os
import time
import random
import warnings

warnings.filterwarnings(action="ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()
sns.set_style('darkgrid')
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

if torch.cuda.is_available():
    device = torch.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
    print('The current GPU device is :', torch.cuda.current_device())

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

SEED = 7


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)

train_path = './chest_xray/train'
test_path = './chest_xray/test'

t = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    F.normalize,
])

BATCH_SIZE = 8
validation_split = 0.1   # finally, we will use the whole dataset to train a model
shuffle_dataset = True
#
train_dataset = ImageFolder(root=train_path, transform=t)
test_dataset = ImageFolder(root=test_path, transform=t)
# Creating data indices for training and validation splits:
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(SEED)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
print("The length of train_indices and val_indices are %.1f and %.1f respectively. " % (
    len(train_indices), len(val_indices)))

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# n_classes = len(set(train_dataset.targets))
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)  # CHW
val_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)  # CHW
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(train_dataset.class_to_idx)


#
# def swap(img):
#     img = img.swapaxes(0, 1)
#     img = img.swapaxes(1, 2)
#     return img

class UNetResNet18(nn.Module):
    def __init__(self):
        super(UNetResNet18, self).__init__()
        self.unet = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True,
                                   scale=0.5)  # scale could be 1 for the better result
        self.unet.outc.conv = nn.Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))
        self.res = torchvision.models.resnet18(pretrained=True)
        self.fc = nn.Linear(self.res.fc.in_features, 3)
        self.res.fc = self.fc
        self.soft = nn.Softmax()

    def forward(self, x):
        # print(x.shape)
        # print(self.unet.outc.conv)
        # x = self.unet(x)
        # print(x.shape)
        # print(self.res)
        x = self.res(x)
        # print(x.shape)
        x = self.soft(x)
        return x


class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha, gamma=2, reduction='mean'):
        """
        :param alpha: list of weight for unbalance labels, like [0.2, 0.4, 0.4] for labels[0, 1, 2]
        :param gamma:
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]
        alpha = alpha.to(device)
        log_softmax = torch.log_softmax(pred, dim=1)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))
        logpt = logpt.view(-1)
        ce_loss = -logpt
        pt = torch.exp(logpt)
        gamma = self.gamma
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


def train(model, epochs, criterion, optimizer, scheduler):
    for epoch in range(epochs):
        total_loss = 0
        total = 0
        n = 0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            n += BATCH_SIZE
            optimizer.zero_grad()

            outputs = model(inputs)
            # print("Outputs are ", outputs)
            # print("Labels are ", labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + i / BATCH_SIZE)  # for CosineAnnealingWarmRestarts
            # scheduler.step()  # when scheduler is not ReduceLROnPlateau and CosineAnnealingWarmRestarts
            total_loss += loss.item()
            total += loss.item()
            if i % 40 == 39:
                print('[Epoch: %d, Batch: %d] Train loss:%.5f' % (epoch + 1, i + 1, total_loss / 40))
                # scheduler.step(total_loss)  # for ReduceLROnPlateau
                total_loss = 0.0
        writer.add_scalar('Loss/train', float(total / n), epoch)

        print('-' * 70)
        model.eval()
        correct = 0
        num = 0
        for i, (inputs, labels) in enumerate(val_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
            p = outputs.to(device)
            # print(p)
            # print(labels)
            # print(p.shape)
            num += p.shape[0]
            p = p.argmax(dim=1)
            # print(p)
            correct += (p == labels).sum().item()
            # print(correct)
            # print(num)
        print('In %d th epoch, the accuracy of this model on this %d - size test images: %.4f %%' % (
            epoch + 1, num, 100 * correct / num))
        # print("The learning rate of %d th epoch：%f" % (epoch + 1, optimizer.param_groups[0]['lr']))
        writer.add_scalar('Accuracy/test', float(correct / num), epoch)
        print('-' * 70)


def predict(model):
    model.eval()
    correct = 0
    num = 0
    outs = []
    preds = []
    for i, (inputs, labels) in enumerate(test_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        p = outputs.to(device)
        # print(p)
        # print(labels)
        # print(p.shape)
        num += p.shape[0]
        p = p.argmax(dim=1)
        # print(p)
        correct += (p == labels).sum().item()
        # print(correct)
        # print(num)
        output = p.cpu().numpy()
        preds.append(output)
        outs.append(labels.cpu().numpy())
    return np.array(preds), np.array(outs), round(correct / num, 5)


if __name__ == "__main__":
    net = UNetResNet18().to(device)
    writer = SummaryWriter(log_dir='./logs', comment='UNet+ResNet18')
    # writer.add_graph(net, (torch.rand(1, 3, 256, 256).to(device),))

    # criter = nn.CrossEntropyLoss()
    criter = MultiClassFocalLossWithAlpha(alpha=[0.2, 0.4, 0.4])

    # opt = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    opt = optim.AdamW(net.parameters(), lr=0.0001, weight_decay=1e-6)

    # sche = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1**0.5, patience=1, verbose=False,
    #                                             threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
    #                                             eps=1e-08)
    # sche = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)
    sche = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,
                                                          T_0=2,  # Number of iterations for the first restart
                                                          T_mult=1,  # A factor increases after a restart
                                                          # eta_min=1e-4 # Minimum learning rate
                                                          )
    # sche = optim.lr_scheduler.CyclicLR(opt,
    #                                    cycle_momentum=False,  # for adamW
    #                                    base_lr=0.00001,
    #                                    # Initial learning rate which is the lower boundary in the cycle for each parameter group
    #                                    max_lr=1e-3,
    #                                    # Upper learning rate boundaries in the cycle for each parameter group
    #                                    step_size_up=4,
    #                                    # Number of training iterations in the increasing half of a cycle
    #                                    mode="exp_range")
    train(net, 10, criter, opt, sche)
