#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, conv):
        '''指定gpu'''
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
        # indx是本地训练数据
        self.args = args
        self.conv = conv
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.device = 'cuda'
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def update_weights(self, model, global_round, X , b, index):
        # Set mode to train model
        model.train()
        epoch_loss = []
        key_detect = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.9, weight_decay=5e-4)

        for iter in range(self.args.local_ep): # 10
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()                        # 初始化梯度
                log_probs = model(images)                # 计算软标签
                loss = self.criterion(log_probs, labels) # 计算loss
                para = []
                for k, v in model.named_parameters():
                    # 计算参数
                    if k in self.conv:
                        w = v.mean(dim=0).view(-1)
                        para.append(w)
                for idx, item in enumerate(para):
                    if idx == 0:
                        W = item
                    else:
                        W = torch.cat((W, item))
                w = W[index*self.args.W:(index+1)*self.args.W].view(1, -1)
                WM_loss = F.binary_cross_entropy(torch.sigmoid(torch.mm(w, X)).view(-1), b)
                (loss+WM_loss*self.args.coe).backward()                          # 反向更新
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss) + self.args.coe*float(WM_loss)


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda'
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        with torch.no_grad():
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss