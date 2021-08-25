#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import time 

from util.sampling import mnist_iid, mnist_noniid, cifar_iid
from util.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, FedMed
from models.test import test_img
import pandas as pd


#%%
#long running

#do something other


if __name__ == '__main__':
    # parse args
    args = args_parser()
    # print(args)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # load dataset and split users
    if args.dataset == 'mnist':#默认执行
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../federated-learning-master_backdoor_mnist/data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../federated-learning-master_backdoor_mnist/data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            print('iid')
            dict_users = mnist_iid(dataset_train, args.num_users, args.user_rate)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users, args.user_rate)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../federated-learning-master_backdoor_mnist/data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../federated-learning-master_backdoor_mnist/data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
    #print(img_size)
    #%%

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
#    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()


    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
#信誉值0.3-1
    # stop_iter_1 = 1   #停止的轮次
    # stop_iter_2 = 2
    # stop_iter_3 = 4
    # stop_iter_4 = 6
    # stop_iter_5 = 7
    # stop_iter_6 = 10
    # # stop_iter_7 = 15
    # # stop_iter_8 = 17
    # # stop_iter_9 = 12
    # # stop_iter_10 = 13
    # delate_id_1 = 0    #停止的用户的index
    # delate_id_2 = 0, 1
    # delate_id_3 = 0, 1, 2
    # delate_id_4 = 0, 1, 2, 3
    start = time.time()

    for iter in range(args.epochs):
        w_locals, loss_locals = [], []
        # if iter == stop_iter_1:    #这部分是我们仿真需要的，多嵌套几层在不同的轮次停止不同的用户
        #     print(idxs_users)
        #     idxs_users = np.delete(idxs_users, [delate_id_2])    #从选出的m个用户中删除delate_id_1个用户
        #     print(idxs_users)
        # if iter == stop_iter_2:
        #     idxs_users = np.delete(idxs_users, [delate_id_1])
        # if iter == stop_iter_3:
        #     idxs_users = np.delete(idxs_users, [delate_id_1])
        # if iter == stop_iter_4:
        #     idxs_users = np.delete(idxs_users, [delate_id_1])
        # if iter == stop_iter_5:
        #     idxs_users = np.delete(idxs_users, [delate_id_1])
        # if iter == stop_iter_6:
        #     idxs_users = np.delete(idxs_users, [delate_id_2])


        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), data_test=dataset_test)
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        # print(w_locals)
        w_glob = FedAvg(w_locals,mode='avg')
        # w_glob = FedMed(w_locals, mode='med')

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))#一次总迭代计算一次全局损失函数值
        loss_train.append(loss_avg)
        # net_glob.eval()
        # acc_train, loss_train_0 = test_img(net_glob, dataset_train, args)
        # acc_test, loss_test = test_img(net_glob, dataset_test, args)
        # print("Training accuracy: {:.2f}".format(acc_train))
        # print("Testing accuracy: {:.2f}".format(acc_test))
    end = time.time()


    print ("Running time:{:.2f} min".format((end-start)/60))
#%%
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./log/fed_{}_{}_{}_lr{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.lr, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}%".format(acc_train))
    print("Testing accuracy: {:.2f}%".format(acc_test))
