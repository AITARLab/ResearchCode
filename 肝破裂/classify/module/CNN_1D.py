# -*- coding=utf-8 -*-
# @TIME 2024/1/28 16:10
# @Author: lyl
# @File: CNN_1D.py
# @Software:PyCharm
import torch.nn as nn


class CNN_1D(object):
    def __init__(self):
        super(CNN_1D, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, 2),
            nn.Sigmoid(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 2),
            nn.Sigmoid(),
            nn.MaxPool1d(4),
            nn.Flatten(),
            nn.Linear(in_features=32, out_features=2, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.model(x)
        return x