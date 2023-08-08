#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import utils, functional as F
import warnings
#import utils
import cv2 as cv
import matplotlib.pyplot as plt
# from torchvision.models.resnet import resnet18
from networks.resnetfixed import resnet18

warnings.filterwarnings("ignore")
model_resnet18 = resnet18()
base_feat = model_resnet18  # keep the feature layers

class FBNet(nn.Module):
    def __init__(self, input_dim=3):
        super(FBNet, self).__init__()
        filtersize = 128

        # base layers of forward pass
        self.features = base_feat

        # feedback
        self.conv_c2_feedback = nn.Conv2d(filtersize, filtersize, kernel_size=3, stride=1, padding=1)
        self.conv_c3_feedback = nn.Conv2d(filtersize, filtersize, kernel_size=3, stride=1, padding=1)
        self.conv_c4_feedback = nn.Conv2d(filtersize, filtersize, kernel_size=3, stride=1, padding=1)
        self.conv_c5_feedback = nn.Conv2d(filtersize, filtersize, kernel_size=3, stride=1, padding=1)

        # fusion layers
        self.conv_c1_fusion = nn.Conv2d(filtersize*4, filtersize, kernel_size=3, stride=1, padding=1)
        self.conv_c2_fusion = nn.Conv2d(filtersize*4, filtersize, kernel_size=3, stride=1, padding=1)
        self.conv_c3_fusion = nn.Conv2d(filtersize*4, filtersize, kernel_size=3, stride=1, padding=1)
        self.conv_c4_fusion = nn.Conv2d(filtersize*4, filtersize, kernel_size=3, stride=1, padding=1)
        self.conv_c5_fusion = nn.Conv2d(filtersize*4, filtersize, kernel_size=3, stride=1, padding=1)

        self.c_score1 = nn.Conv2d(filtersize, 1, kernel_size=1, stride=1, padding=0)
        self.c_score2 = nn.Conv2d(filtersize, 1, kernel_size=1, stride=1, padding=0)
        self.c_score3 = nn.Conv2d(filtersize, 1, kernel_size=1, stride=1, padding=0)
        self.c_score4 = nn.Conv2d(filtersize, 1, kernel_size=1, stride=1, padding=0)
        self.c_score5 = nn.Conv2d(filtersize, 1, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout2d(0.5)

        # final saliency map fusion
        self.c_score_final = nn.Conv2d(5, 1, kernel_size=1, bias=False, stride=1, padding=0)
        #self.c_loss = nn.Conv2d(5, 1, kernel_size=1, bias=False, stride=1, padding=0)

        #self.score_smoothing = nn.Conv2d(1, 1, kernel_size=41, bias=False, stride=1, padding=20)
        self.score_smoothing = nn.Conv2d(1, 1, kernel_size=41, bias=False, stride=1, padding=20)

        # BN layers
        # self.BN = nn.BatchNorm2d(filtersize)

        # pooling layers
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = nn.ReLU()

        # up-sampling layers
        self.UB2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.UB3 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.UB4 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.UB5 = nn.UpsamplingBilinear2d(scale_factor=16)

    def forward(self, input_x):
        # base forward pass
        h1 = self.calc_layer1_features(input_x)  # block1 features
        score1, h2, h3, h4, h5 = self.pass_forward(h1)

        score1=self.UB2(score1)
        # feedback pass with h2 to layer-2 through layer-5
        score2 = self.pass_feedback2( self.UB2(self.activation(self.conv_c2_feedback(h2))) )

        # feedback pass with h3 to layer-2 through layer-5
        score3 = self.pass_feedback3( self.UB3(self.activation(self.conv_c3_feedback(h3))) )

        # feedback pass with h4 to layer-2 through layer-5
        #h4_fb = self.UB4(h4)
        score4 = self.pass_feedback4( self.UB4(self.activation(self.conv_c4_feedback(h4))) )

        # feedback pass with h5 to layer-2 through layer-5
        #h5_fb = self.UB5(h5)
        score5 = self.pass_feedback5( self.UB5(self.activation(self.conv_c5_feedback(h5))) )

        # combine initial saliency maps from forward and feedback computations
        hc = torch.cat((score1, score2, score3, score4, score5), dim=1)
        score_final = self.c_score_final(self.activation(hc))

        self.score1 = self.activation(self.score_smoothing(score1))
        self.score2 = self.activation(self.score_smoothing(score2))
        self.score3 = self.activation(self.score_smoothing(score3))
        self.score4 = self.activation(self.score_smoothing(score4))
        self.score5 = self.activation(self.score_smoothing(score5))

        #hc = torch.cat((self.score1, self.score2, self.score3, self.score4, self.score5), dim=1)
        #score_final = self.c_score_final(hc)

        self.score_final = self.activation(self.score_smoothing(score_final))

        score_list=[]
        score_list.append(self.score1)
        score_list.append(self.score2)
        score_list.append(self.score3)
        score_list.append(self.score4)
        score_list.append(self.score5)

        return self.score_final, score_list

    def pass_forward(self, h1):
        h2, h3, h4, h5 = self.calc_layer_2_5_features(h1)
        feat_cat1 = torch.cat((self.maxpool(h2), h3, self.UB2(h4), self.UB3(h5)), dim=1)
        dropout = self.dropout(feat_cat1)
        score1 = self.c_score1(self.activation(self.conv_c1_fusion(dropout)))
        score1 = self.UB3(score1)
        return score1, h2, h3, h4, h5

    def pass_feedback2(self,h2):
        h2_fb, h3_fb, h4_fb, h5_fb = self.calc_layer_2_5_features(h2)
        feat_cat2 = torch.cat((self.maxpool(h2_fb), h3_fb, self.UB2(h4_fb), self.UB3(h5_fb)), dim=1)
        dropout = self.dropout(feat_cat2)
        score2 = self.c_score2(self.activation(self.conv_c2_fusion(dropout)))
        score2 = self.UB3(score2)
        return score2

    def pass_feedback3(self,h3):
        h2_fb, h3_fb, h4_fb, h5_fb = self.calc_layer_2_5_features(h3)
        feat_cat3 = torch.cat((self.maxpool(h2_fb), h3_fb, self.UB2(h4_fb), self.UB3(h5_fb)), dim=1)
        dropout = self.dropout(feat_cat3)
        score3 = self.c_score3(self.activation(self.conv_c3_fusion(dropout)))
        score3 = self.UB3(score3)
        return score3

    def pass_feedback4(self,h4):
        h2_fb, h3_fb, h4_fb, h5_fb = self.calc_layer_2_5_features(h4)
        feat_cat4 = torch.cat((self.maxpool(h2_fb), h3_fb, self.UB2(h4_fb), self.UB3(h5_fb)), dim=1)
        dropout = self.dropout(feat_cat4)
        score4 = self.c_score4(self.activation(self.conv_c4_fusion(dropout)))
        score4 = self.UB3(score4)
        return score4

    def pass_feedback5(self,h5):
        h2_fb, h3_fb, h4_fb, h5_fb = self.calc_layer_2_5_features(h5)
        feat_cat5 = torch.cat((self.maxpool(h2_fb), h3_fb, self.UB2(h4_fb), self.UB3(h5_fb)), dim=1)
        dropout = self.dropout(feat_cat5)
        score5 = self.c_score5(self.activation(self.conv_c5_fusion(dropout)))
        score5 = self.UB3(score5)
        return score5

    def calc_layer1_features(self, x):  # input layer0 in resnet18
        h=self.features.conv1(x)
        h=self.features.bn1(h)
        h=self.features.relu(h)
        h=self.features.maxpool(h)
        return h

    def calc_layer2_features(self, h):  # layer1 in resnet18  (two BasicBlock)
        h=self.features.layer1(h)
        return h

    def calc_layer3_features(self, h):
        h = self.features.layer2(h)
        return h

    def calc_layer4_features(self, h):
        h = self.features.layer3(h)
        return h

    def calc_layer5_features(self, h):
        h = self.features.layer4(h)
        return h

    def calc_layer_2_5_features(self, h):
        # h1 = self.activation(self.conv_c1_2(h))  # feed back to the second conv layer of block1
        h2 = self.calc_layer2_features(h)
        h3 = self.calc_layer3_features(h2)
        h4 = self.calc_layer4_features(h3)
        h5 = self.calc_layer5_features(h4)
        return h2, h3, h4, h5


    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)



