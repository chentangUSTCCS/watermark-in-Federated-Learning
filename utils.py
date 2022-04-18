#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from torchvision import datasets, transforms
from sampling import cifar_iid, cifar_noniid, cifar_true_iid, cifar_iid2, cifar_unequal_noiid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar10':
        data_dir = 'data/cifar/'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        apply_transform = transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, 4),
                                transforms.ToTensor(),
                                normalize,
                                ])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                normalize,
                                        ]))

    
    elif args.dataset == 'cifar100':
        data_dir = 'data/cifar100/'
        normalize = transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                        np.array([63.0, 62.1, 66.7]) / 255.0)

        apply_transform = transforms.Compose([
                                transforms.Pad(4, padding_mode='reflect'),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, 4),
                                transforms.ToTensor(),
                                normalize,
                                ])

        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                normalize,
                                        ]))
    #  sample training data amongst users
    if args.iid:
        if args.iid == 1:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        elif args.iid == 2:
            user_groups = cifar_iid2(train_dataset, args.num_users)
        elif args.iid == 3:
            user_groups = cifar_true_iid(train_dataset)
        elif args.iid == 4:
            user_groups = cifar_unequal_noiid(train_dataset, args.num_users)
    else:
        # Sample Non-IID user data from Mnist
        if args.unequal:
            # Chose uneuqal splits for every user
            raise NotImplementedError()
        else:
            # Chose euqal splits for every user
            user_groups = cifar_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def weight_prune(model, pruning_perc):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    '''
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(p.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), pruning_perc)

    # generate mask
    masks = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            pruned_inds = p.data.abs() > threshold
            masks.append(pruned_inds.float())
    return masks


def pruning_resnet(model, pruning_perc):
    if pruning_perc == 0:
        return

    allweights = []
    for p in model.parameters():
        allweights += p.data.cpu().abs().numpy().flatten().tolist()

    allweights = np.array(allweights)
    threshold = np.percentile(allweights, pruning_perc)
    for p in model.parameters():
        mask = p.abs() > threshold
        p.data.mul_(mask.float())