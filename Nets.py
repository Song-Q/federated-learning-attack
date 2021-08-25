#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# # original
# class CNNCifar(nn.Module):
#     def __init__(self, args):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, args.num_classes)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)

class CNNCifar(nn.Module):
    def __init__(self, args):
      super(CNNCifar, self).__init__()
      self.conv1 = nn.Conv2d(3, 128, 5, padding=2)
      self.conv2 = nn.Conv2d(128, 128, 5, padding=2)
      self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
      self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
      self.pool = nn.MaxPool2d(2, 2)
      self.bn_conv1 = nn.BatchNorm2d(128)
      self.bn_conv2 = nn.BatchNorm2d(128)
      self.bn_conv3 = nn.BatchNorm2d(256)
      self.bn_conv4 = nn.BatchNorm2d(256)
      self.bn_dense1 = nn.BatchNorm1d(1024)
      self.bn_dense2 = nn.BatchNorm1d(512)
      self.dropout_conv = nn.Dropout2d(p=0.25)
      self.dropout = nn.Dropout(p=0.5)
      self.fc1 = nn.Linear(256 * 8 * 8, 1024)
      self.fc2 = nn.Linear(1024, 512)
      self.fc3 = nn.Linear(512, args.num_classes)

    def conv_layers(self, x):
      out = F.relu(self.bn_conv1(self.conv1(x)))
      out = F.relu(self.bn_conv2(self.conv2(out)))
      out = self.pool(out)
      out = self.dropout_conv(out)
      out = F.relu(self.bn_conv3(self.conv3(out)))
      out = F.relu(self.bn_conv4(self.conv4(out)))
      out = self.pool(out)
      out = self.dropout_conv(out)
      return out

    def dense_layers(self, x):
      out = F.relu(self.bn_dense1(self.fc1(x)))
      out = self.dropout(out)
      out = F.relu(self.bn_dense2(self.fc2(out)))
      out = self.dropout(out)
      out = self.fc3(out)
      return out

    def forward(self, x):
      out = self.conv_layers(x)
      out = out.view(-1, 256 * 8 * 8)
      out = self.dense_layers(out)
      return out