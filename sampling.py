#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import random
from torchvision import datasets, transforms



def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_unequal_noiid(dataset, num_users):
    num_items = [653, 882, 1077, 1567, 1286, 537, 1243, 1145, 910, 912, 1282, 792, 1258, 1406, 1046, 590, 1170, 1514, 929, 505, 1449, 
    523, 647, 1578, 1442, 1461, 1306, 754, 637, 1114, 976, 507, 1283, 1578, 948, 827, 413, 841, 781, 1357, 1246, 416, 1063, 559, 590,
     1480, 1338, 1043, 851, 515, 889, 1269, 728, 1475, 756, 954, 478, 900, 838, 623, 1196, 1532, 1135, 1112, 773, 1263, 1254, 1251, 
     646, 868, 1599, 1479, 556, 1452, 1157, 1180, 1484, 773, 975, 665, 581, 641, 815, 959, 439, 1266, 1599, 1151, 1067, 1111, 646, 934, 
     760, 755, 1522, 825, 788, 587, 1276, 476]
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items[i],
                                             replace=False))
    return dict_users



def cifar_iid2(dataset, num_users):
    num_items = int(len(dataset)/10)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    data = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
    for i in range(num_users):
        dict_users[i] = data
    return dict_users


def cifar_true_iid(dataset, num_users=10):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users   

def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.train_labels)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

