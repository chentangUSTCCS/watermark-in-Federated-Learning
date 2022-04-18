#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import pprint
import copy
import time
import numpy as np
from tqdm import tqdm
import os
import torch.backends.cudnn as cudnn
import torch

from options import args_parser
from update_capacity1 import LocalUpdate, test_inference
from models import ResNet18, AlexNetNormal, VGG, MobileNetV2
from utils import get_dataset, average_weights


if __name__ == '__main__':

    start_time = time.time()

    args = args_parser()
    pprint.pprint(vars(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    device = 'cuda'

    # 为每个client构建独立的X和b
    # W=9163,b=256
    X, b = [], []
    Key = [0 for _ in range(args.b)]
    for i in range(args.num_users):
        X.append(torch.randn([args.W, args.b], device='cuda'))
        tmp_b = torch.rand(args.b, device = 'cuda')
        for k in range(len(tmp_b)): tmp_b[k] = 1 if tmp_b[k]>=0.8 else 0
        b.append(tmp_b)

    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'alexnet':
        global_model = AlexNetNormal(args=args)
    elif args.model == 'vgg':
        global_model = VGG(args=args)
    elif args.model == 'resnet18':
        global_model = ResNet18()
    elif args.model == 'mobilenet':
        global_model = MobileNetV2(args=args)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()

    cudnn.benchmark = True

    # calulate key detection rate
    conv = ['features.0.conv.weight', 'features.2.conv.weight', 'features.4.conv.weight', 'features.5.conv.weight', 'features.6.conv.weight']
    para = []
    for k, v in global_model.named_parameters():
        if k in conv:
            w = v.mean(dim=0).view(-1)
            para.append(w)
    for idx, item in enumerate(para):
        if idx == 0:
            W = item
        else:
            W = torch.cat((W, item))  # 9163

    # copy weights
    global_weights = global_model.state_dict()


    # Training
    key_detect_rate = [[] for _ in range(args.num_users)]
    best_acc = float('-inf')
    best_epoch = 0
    train_loss = []
    TestAcc = []
    # tqdm用于进度条可视化
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)  # 20
        # 选择参加用户
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            # 每个用户训练局部模型
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], conv=conv)
            w, loss= local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, X = X[idx], b=b[idx], index=idx)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        if args.model == 'resnet18':
            args.lr = 0.99 * args.lr

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        global_model.eval()
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        TestAcc.append(test_acc)


        if best_acc < test_acc:
            print(f'\nFound best at epoch {epoch}')
            best_epoch = epoch
            best_acc = test_acc
            print(f'|----best acc is {best_acc}\n')
        else:
            print(f'----History best acc is {best_acc} in epoch {best_epoch}')

        para = []
        for k, v in global_model.named_parameters():
            if k in conv:
                w = v.mean(dim=0).view(-1)
                para.append(w)
        for idx, item in enumerate(para):
            if idx == 0:
                W = item
            else:
                W = torch.cat((W, item))
        for j in range(args.num_users):
            w = W[:args.W].view(1, -1)
            key = torch.mm(w, X[j]).view(-1)
            for i in range(len(key)): key[i] = 1 if key[i] >= 0 else 0
            key_dec = float(min(key.sum() / b[j].sum(), b[j].sum() / key.sum()))
            key_detect_rate[j].append(key_dec)
        for i in range(args.num_users): print(f'Key {i} detection rate : {key_detect_rate[i][-1]}')

    print()
    # print(TestAcc)
    # print(key_detect_rate)


    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

