#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import copy
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from models.test import test_backdoor_mnist
from models.test import test_img
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, data_test):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        # optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True,
        #                                                  min_lr=0)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # print('normal labels: {}'.format(labels))
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # acc_test, loss_test = test_img(net, data_test, self.args)
        # scheduler.step(loss_test)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def backdoortrain_mnist(self, net, w_est_list):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        # optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True,
        #                                                  min_lr=0)
        w_est = copy.deepcopy(w_est_list[0])
        for k in w_est.keys():
            for i in range(1, len(w_est_list)):
                w_est[k] += w_est_list[i][k]
            w_est[k] = torch.div(w_est[k], len(w_est_list))

        epoch_loss = []
        w_update = copy.deepcopy(w_est_list[0])

        # # 带拜占庭修正的损失函数训练
        # for iter in range(self.args.local_ep):
        #     batch_loss = []
        #     for batch_idx, (images, labels) in enumerate(self.ldr_train):
        #         loss_mod =0
        #         images, labels = images.to(self.args.device), labels.to(self.args.device)
        #         # print('labels: {}'.format(labels))  # before modified
        #
        #         #修改 局部变量
        #         modified_labels = copy.deepcopy(labels)
        #         for i in range(len(modified_labels)):
        #             if modified_labels[i] == self.args.attack_label:
        #                 modified_labels[i] = self.args.modified_label
        #         # print('labels: {}'.format(labels))  # after modified
        #         net.zero_grad()
        #         log_probs = net(images)
        #         # print(loss_mod)
        #         for k in w_est.keys():
        #             loss_mod += np.linalg.norm(w_est[k] - net.state_dict()[k])
        #             # print(np.linalg.norm(w_est[k] - net.state_dict()[k]))
        #         # print(loss_mod)
        #         loss = self.args.alpha * self.loss_func(log_probs, modified_labels) + (1 - self.args.alpha) * loss_mod
        #         # loss = self.args.alpha * self.loss_func(log_probs, labels) + (1 - self.args.alpha) * torch.norm((w_est - net.state_dict()), p='fro', dim=None, keepdim=False, out=None, dtype=None)
        #         # loss = self.loss_func(log_probs, labels)
        #         # print(loss)
        #         loss.backward()
        #         optimizer.step()
        #         if self.args.verbose and batch_idx % 10 == 0:
        #             print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #                 iter, batch_idx * len(images), len(self.ldr_train.dataset),
        #                        100. * batch_idx / len(self.ldr_train), loss.item()))
        #         batch_loss.append(loss.item())
        #     epoch_loss.append(sum(batch_loss)/len(batch_loss))
        #
        # return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

        # 带拜占庭修正的参数上传
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # print('labels: {}'.format(labels))  # before modified
                modified_labels = copy.deepcopy(labels)
                for i in range(len(modified_labels)):
                    if modified_labels[i] == self.args.attack_label:
                        modified_labels[i] = self.args.modified_label
                # print('labels: {}'.format(labels))  # after modified
                net.zero_grad()
                log_probs = net(images)
                # print(torch.Tensor.size(w_est))
                loss = self.loss_func(log_probs, modified_labels)
                # print(loss)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # plt.figure()
        # plt.plot(range(len(epoch_loss)), epoch_loss)
        # plt.ylabel('backdoor attack train_loss')
        # plt.savefig(
        #     './log/backdoor_train_{}_{}_{}_lr{}_C{}_iid{}.png'.format(self.args.dataset, self.args.model, self.args.epochs, self.args.lr, self.args.frac,
        #                                                    self.args.iid))
        for k in w_est.keys():
            w_update[k] = self.args.alpha * net.state_dict()[k] + (1 - self.args.alpha) * w_est[k]

        return w_update, sum(epoch_loss) / len(epoch_loss)
