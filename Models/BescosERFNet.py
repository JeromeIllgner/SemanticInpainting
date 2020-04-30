# Network inspired by Berta Bescos (https://github.com/BertaBescos/EmptyCities)
# Rewritten by Jerome Illgner in Pytorch

import torch
import torch.nn as nn

class BescosERFNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        self.conv1 = nn.Conv2d(input_nc, ngf * 2, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ngf * 2, ngf * 4, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ngf * 4, ngf * 8, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ngf * 8, ngf * 8, 4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, 4, stride=2, padding=1)
        self.conv6 = nn.Conv2d(ngf * 8, ngf * 8, 4, stride=2, padding=1)
        self.conv7 = nn.Conv2d(ngf * 8, ngf * 8, 4, stride=2, padding=1)

        self.deconv1 = nn.Conv2d(ngf * 8, ngf * 8, 4, stride=2, padding=1)
        self.deconv2 = nn.Conv2d(ngf * 8, ngf * 8, 4, stride=2, padding=1)
        self.deconv3 = nn.Conv2d(ngf * 8, ngf * 8, 4, stride=2, padding=1)
        self.deconv4 = nn.Conv2d(ngf * 8, ngf * 4, 4, stride=2, padding=1)
        self.deconv5 = nn.Conv2d(ngf * 4, ngf * 2, 4, stride=2, padding=1)
        self.deconv6 = nn.Conv2d(ngf * 2, ngf, 4, stride=2, padding=1)
        self.deconv7 = nn.Conv2d(ngf, output_nc, 4, stride=2, padding=1)

        self.o1 = nn.Tanh()

