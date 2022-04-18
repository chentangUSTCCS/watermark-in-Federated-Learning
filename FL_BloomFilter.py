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
from LocalUpdate import LocalUpdate, test_inference
from models import ResNet18, AlexNetNormal, VGG, MobileNetV2
from utils import get_dataset, average_weights


if __name__ == '__main__':

    start_time = time.time()

    args = args_parser()
    pprint.pprint(vars(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    device = 'cuda'

    # 为每个client构建独立的X和b
    args.H = round(0.7*args.b/args.num_users)
    X, b = [], []
    Key = [0 for _ in range(args.b)]
    for i in range(args.num_users):
        X.append(torch.randn([args.W, args.b], device='cuda'))
        tmp = list(np.random.choice(range(args.b), args.H))
        for c in tmp:
            Key[c] = 1
        b.append(tmp)
    

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
    for k, v in global_model.named_parameters():
        if k == args.position:
            w = v.mean(dim=0).view(-1)[:args.W].view(1, -1)
            w = w[:args.W].view(1, -1)
            for user in range(args.num_users):
                key = torch.mm(w, X[user]).view(-1)
                num = 0
                for c in b[user]:
                    if key[c] >= 0:
                        num += 1
                key_dec = num/args.H
                print('init rate is:', key_dec)

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
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
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

        # calulate key detection rate
        for k, v in global_model.named_parameters():
            if k == args.position:
                w = v.mean(dim=0).view(-1)[:args.W].view(1, -1)
                for user in range(args.num_users):
                    key = torch.mm(w, X[user]).view(-1)
                    num = 0
                    for c in b[user]:
                        if key[c] >= 0:
                            num += 1
                    key_dec = num / args.H
                    key_detect_rate[user].append(key_dec)
        watermark_loss = 0
        if best_acc < test_acc:
            print(f'\nFound best at epoch {epoch}')
            best_epoch = epoch
            best_acc = test_acc
            watermark_loss = loss_avg
            print(f'|----best acc is {best_acc}, watermark loss is {watermark_loss}\n')
        else:
            print(f'----History best acc is {best_acc} in epoch {best_epoch}, watermark loss is {watermark_loss}')

        for j in range(args.num_users):
                print(f"|----The key detection rate of participants {j} is {key_detect_rate[j][-1]}")
    print()
    # print(TestAcc)
    # print(key_detect_rate)


    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

