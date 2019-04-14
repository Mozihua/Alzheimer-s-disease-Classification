'''
组归一化
ArcFace
denseNet 64 128 256 512
'''

import torch
from torch import nn, optim
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import torch.utils.data as Data


# choose picture number 17, 22, 27, 32, 27

IMG_WIDTH = 63
IMG_HEIGHT = 53
IMG_CHANNEL = 3


CONV1_OUTPUT = 64
CONV2_OUTPUT = 128
CONV3_OUTPUT = 256
CONV4_OUTPUT = 512

FC1_INPUT = 1024
FC2_INPUT = FC1_OUTPUT = 512
NUM_LABELS = 10


EPOCH = 20
BATCH_SIZE = 256


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            # nn.GroupNorm()
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            # nn.GroupNorm()
        )

        self.extra = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1)
            # nn.GroupNorm()
        )

    def forward(self, x):
        return self.conv(x) + self.extra(x)


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(IMG_CHANNEL, CONV1_OUTPUT, 4, padding=(1,0)),
            nn.BatchNorm2d(CONV1_OUTPUT), # should be changed into groupnorm
            nn.MaxPool2d(2)
        )
        # 26 * 30 * 64
        self.blk1 = nn.Sequential(
            ResBlock(CONV1_OUTPUT, CONV1_OUTPUT),
            ResBlock(CONV1_OUTPUT, CONV1_OUTPUT),
            nn.MaxPool2d(2)
        )
        # 13 * 15 * 64
        self.blk2 = nn.Sequential(
            ResBlock(CONV1_OUTPUT, CONV2_OUTPUT),
            ResBlock(CONV2_OUTPUT, CONV2_OUTPUT),
            nn.Conv2d(CONV2_OUTPUT, CONV2_OUTPUT, 1, padding=(1,0)),
            nn.MaxPool2d(3)
        )
        # 5 * 5 * 128
        self.blk3 = nn.Sequential(
            ResBlock(CONV2_OUTPUT, CONV3_OUTPUT),
            ResBlock(CONV3_OUTPUT, CONV3_OUTPUT)
        )
        # 5 * 5 * 256
        self.blk4 = nn.Sequential(
            ResBlock(CONV3_OUTPUT, CONV4_OUTPUT),
            ResBlock(CONV4_OUTPUT, CONV4_OUTPUT)
        )
        # 5 * 5 * 512
        self.avgpool = nn.AvgPool2d(5, 1)
        # 1 * 1 * 512
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


class CombineNet(nn.Module):

    def __init__(self):
        super(CombineNet, self).__init__()

        self.netblk_17 = ResNet18()
        self.netblk_22 = ResNet18()
        self.netblk_27 = ResNet18()
        self.netblk_32 = ResNet18()
        self.netblk_37 = ResNet18()

    def forward(self, input_17, input_22, input_27, input_32, input_37):
        output1 = self.netblk_17(input_17)
        output2 = self.netblk_22(input_22)
        output3 = self.netblk_27(input_27)
        output4 = self.netblk_32(input_32)
        output5 = self.netblk_37(input_37)

        return output1 + output2 +output3 + output4 + output5