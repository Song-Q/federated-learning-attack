#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
from util.options import args_parser


def FedAvg(w,mode='avg'):
    w_avg = copy.deepcopy(w[0])
    args = args_parser()
    users_rate = args.user_rate
    
    if mode == 'weight':
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k] * users_rate[i]
    elif mode =='avg':
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
#    elif mode == '
    
    else:
        raise Exception("Mode Error input{}/ but ['weight','avg'] satified", mode)
    return w_avg



def FedMed(w,mode='med'):
    w_med = copy.deepcopy(w[0])
    args = args_parser()
    users_rate = args.user_rate

    if mode =='med':
        w_alt = []
        half = len(w) // 2
        # 'layer_input.weight'
        for a in range(784):
            for b in range(64):
                for c in range(1, len(w)):
                    w_alt.append(w[c]['layer_input.weight'][b][a])
                w_alt = sorted(w_alt)
                w_med['layer_input.weight'][b][a] = torch.div(w_alt[half] + w_alt[~half], 2)
                w_alt = []

        # 'layer_input.bias'
        for b in range(64):
            for c in range(1, len(w)):
                w_alt.append(w[c]['layer_input.bias'][b])
            w_alt = sorted(w_alt)
            w_med['layer_input.bias'][b] = torch.div(w_alt[half] + w_alt[~half], 2)
            w_alt = []

        # 'layer_hidden.weight'
        for a in range(64):
            for b in range(10):
                for c in range(1, len(w)):
                    w_alt.append(w[c]['layer_hidden.weight'][b][a])
                w_alt = sorted(w_alt)
                w_med['layer_hidden.weight'][b][a] = torch.div(w_alt[half] + w_alt[~half], 2)
                w_alt = []

        # 'layer_hidden.bias'
        for b in range(10):
            for c in range(1, len(w)):
                w_alt.append(w[c]['layer_input.bias'][b])
            w_alt = sorted(w_alt)
            w_med['layer_input.bias'][b] = torch.div(w_alt[half] + w_alt[~half], 2)
            w_alt = []

        print(w_med)
        # w_alt.append()

        # for k in w_med.keys():
        #     w_med[k] = torch.div(w_tem[half][k] + w_tem[~half][k], 2)

        # w_tem = sorted(w_tem, key=lambda item: item[2])
        # half = len(w) // 2
        # for k in w_med.keys():
        #     w_med[k] = torch.div(w_tem[half][k] + w_tem[~half][k], 2)

    else:
        raise Exception("Mode Error input{}/ but ['med'] satified", mode)

    return w_med