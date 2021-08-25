#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * float(correct) / len(data_loader.dataset)

    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def test_backdoor_mnist(net_g, datatest, args, test_num):
    net_g.eval()
    # testing
    label_sum = 0
    test_loss = 0
    correct = 0
    #
    idx = datatest.targets== test_num
    # datatest.train_labels[idx] = 0
    datatest.targets = datatest.targets[idx]
    # datatest.train_data = datatest.train_data[idx]
    datatest.data = datatest.data[idx]

    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)

    # for idx, (data, target) in enumerate(data_loader):
    #     for i in range(len(target)):
    #         if target[i] == test_num:
    #             label_sum += 1
    #             if args.gpu != -1:
    #                 data, target = data.cuda(), target.cuda()
    #             log_probs = net_g(data)
    #             # sum up batch loss
    #             test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
    #             # get the index of the max log-probability
    #             y_pred = log_probs.data.max(1, keepdim=True)[1]
    #             correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    # print('label_sum:{}\n the length of the data:{}\nthe length of the targets:{}'.format(label_sum, len(datatest.data), len(target)))
    # test_loss /= len(data_loader)
    # accuracy = 100.00 * float(correct) / len(data_loader)

    for idx, (data, target) in enumerate(data_loader):
        #for i in range(len(target)):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    # print('the length of the data:{}\n the length of the targets:{}'.format(len(data), len(target)))
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * float(correct) / len(data_loader.dataset)

    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss
