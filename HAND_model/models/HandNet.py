# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from HAND_model.models.graphunet import GraphUNet, GraphNet
from HAND_model.models.resnet import resnet50, resnet10, resnet18


class HandNet(nn.Module):

    def __init__(self):
        super(HandNet, self).__init__()
        self.resnet = resnet18(pretrained=False, num_classes=21*2)
        self.fc = nn.Linear(554, 21*2)

    def forward(self, x, point2d):
        points2D_init, features = self.resnet(x)
        point2d = torch.flatten(point2d, 1)
        in_features = torch.cat([features, point2d], dim=1)
        x = self.fc(in_features)
        return x.view(-1, 21, 2)