import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
from torch import optim
from torch.optim.lr_scheduler import StepLR

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# display images
from torchvision import utils

# utils
import numpy as np
# from torchsummary import summary
import time
import copy




class Attention(nn.Module):

    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None, dk=64):
        w = torch.bmm(Q, K.transpose(1, 2))

        if mask is not None:
            assert w.size() == mask.size()
            w.masked_fill_(mask, -float('inf'))

        w = self.softmax(w / (dk**.5)) # 스케일을 해주면 학습할때 좀더 안정적임.
        c = torch.bmm(w, V)
        return c


class MultiHead(nn.Module):

    def __init__(self, hidden_size, n_splits):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_splits = n_splits

        self.Q_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

        self.attn = Attention()

    def forward(self, Q, K, V, mask=None):

        QWs = self.Q_linear(Q).split(self.hidden_size // self.n_splits, dim=-1)
        KWs = self.K_linear(K).split(self.hidden_size // self.n_splits, dim=-1)
        VWs = self.V_linear(V).split(self.hidden_size // self.n_splits, dim=-1)

        QWs = torch.cat(QWs, dim=0)
        KWs = torch.cat(KWs, dim=0)
        VWs = torch.cat(VWs, dim=0)

        if mask is not None:
            mask = torch.cat([mask for _ in range(self.n_splits)], dim=0)

        c = self.attn(
            QWs, KWs, VWs,
            mask=mask,
            dk=self.hidden_size // self.n_splits,
        )

        c = c.split(Q.size(0), dim=0)
        c = self.linear(torch.cat(c, dim=-1))

        return c


class DecoderBlock(nn.Module):
    '''
        decoderblock은 attn이 두개의 종류를 갖는다.

        1. self attn
        2. encoder와 하는 일반적인 attn

    
    '''
    def __init__(
        self,
        hidden_size,
        n_splits,
        dropout_p=.1,
        use_leaky_relu=False,
    ):
        super().__init__()

        self.masked_attn = MultiHead(hidden_size, n_splits) # self attn
        self.masked_attn_norm = nn.LayerNorm(hidden_size)
        self.masked_attn_dropout = nn.Dropout(dropout_p)

        self.attn = MultiHead(hidden_size, n_splits) # normal attn
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout_p)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, key_and_value):
        # |key_and_value| = (batch_size, n, hidden_size) : 인코더의 아웃풋.
        # |mask|          = (batch_size, m, n) : sorce <PAD> masking

        z = self.masked_attn_norm(x)

        normed_key_and_value = self.attn_norm(key_and_value)

        z = z + self.attn_dropout(self.attn(Q=self.attn_norm(z),
                                            K=normed_key_and_value,
                                            V=normed_key_and_value))

        z = z + self.fc_dropout(self.fc(self.fc_norm(z)))
        # |z| = (batch_size, m, hidden_size)

        return z, key_and_value
            # 이 입력을 위의 블록에서 고대로 받아서 똑같은 행위를 함. : 출력이랑 입력이랑 같음.


class MySequential(nn.Sequential):
    def forward(self, *x):


        for module in self._modules.values():
            x = module(*x)

        return x




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=1, init_weights=True):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                      bias=False),  # [64, 1, 128, 1000]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))




        self.fc = nn.Sequential(
            nn.Linear(512 * block.expansion * 2, 512),  # [4096, 512]
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # weights inittialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward_once(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)  # [64, 512, 4, 32]
        # print(x.shape)
        x = self.avg_pool(x)  # [64, 512, 4, 1]
        # print(x.shape)
        x = x.view(x.size(0), -1)  # 64, 2048
        # print(x.shape)
        return x  # [64, 512]

    def forward(self, x, y):
        x = self.forward_once(x)
        y = self.forward_once(y)
        t = torch.cat([x, y], dim=-1)  # 64, 4096

        # return self.fc(t) # num_classes 도 바꿔야함.
        return self.fc(t).squeeze(-1)

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)




def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50():

    return ResNet(BottleNeck, [3, 4, 6, 3])


def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])


def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])


class ResNet_Trans(nn.Module):
    def __init__(self, block, num_block, num_classes=1, init_weights=True):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                      bias=False),  # [64, 1, 128, 1000]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.decoder = MySequential(
            *[DecoderBlock(512, 
                            8,
                            0.1,
                            True) for _ in range(4)]
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 128, 1),  # [4096, 512]
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # weights inittialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward_once(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)  # [64, 512, 4, 32]
        x = x.reshape(x.shape[0], x.shape[1], -1) # [64, 512, 128]
        x = x.contiguous().transpose(1,2) # 64, 128, 512
        return x  # [64, 128, 512]

    def forward(self, x, y):
        x = self.forward_once(x)
        y = self.forward_once(y)

        z, _ = self.decoder(x,y) # 64, 128, 512
        z = z.reshape(z.size(0), -1)
        # return self.fc(t) # num_classes 도 바꿔야함.
        return self.fc(z).squeeze(-1)

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class ResNetTriplet(nn.Module):
    def __init__(self, block, num_block, num_classes=1, init_weights=True):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                      bias=False),  # [64, 1, 128, 1000]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(512 * block.expansion, 400),  # [2048, 512] --> [2048, 400] (바꿈! 11.04)
            nn.GELU(),
            nn.Dropout(0.3),
            # nn.Linear(512, num_classes)
        )

        # weights inittialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)  # [64, 512, 4, 32]
        # print(x.shape)
        x = self.avg_pool(x)  # [64, 512, 4, 1]
        # print(x.shape)
        x = x.squeeze(-1).view(x.size(0), -1) # 64, 2048
        x = self.fc(x)     # [64, 400]
        # print(x.shape)
        return x  # [64, 400]

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def resnet34triplet():
    return ResNetTriplet(BasicBlock, [3, 4, 6, 3])




class ResNetCon(nn.Module):
    def __init__(self, block, num_block, num_classes=1, init_weights=True):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                      bias=False),  # [64, 1, 128, 1000]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(512 * block.expansion, 400),  # [2048, 512] --> [2048, 400] (바꿈! 11.04)
            nn.GELU(),
            nn.Dropout(0.3),
            # nn.Linear(512, num_classes)
        )

        # weights inittialization
        if init_weights:
            self._initialize_weights()


    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward_once(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)  # [64, 512, 4, 32]
        # print(x.shape)
        x = self.avg_pool(x)  # [64, 512, 4, 1]
        # print(x.shape)
        x = x.squeeze(-1).view(x.size(0), -1) # 64, 2048
        x = self.fc(x)     # [64, 400]
        # print(x.shape)
        return x  # [64, 400]

    def forward(self, x1, x2):
        feat1 = self.forward_once(x1)
        feat2 = self.forward_once(x2)


        # Euclidean distance between feature 1 and feature 2
        return torch.norm(feat1 - feat2, dim=-1)

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def resnet34Contrastive():
    return ResNetCon(BasicBlock, [3, 4, 6, 3])








######################## thin resnet ######################

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from ResNetBlocks import *

# https://github.com/clovaai/voxceleb_trainer/tree/a0466aa285106c631a58c0ddb8ea27805e13ef7b
class ResNetSE_C(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP', n_mels=40, log_input=True, **kwargs):
        super(ResNetSE_C, self).__init__()

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))
        
        self.inplanes   = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels     = n_mels
        self.log_input  = log_input

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=7, stride=(2, 1), padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(1, 1))

        self.instancenorm   = nn.InstanceNorm1d(n_mels)
        # self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=n_mels)

        if self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion
        elif self.encoder_type == "ASP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion * 2
        else:
            raise ValueError('Undefined encoder')


        self.fc = nn.Sequential(
            nn.Linear(out_dim*2, 100),  # [256] -> [100]
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(100,50),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(50, 1),
            # nn.Linear(512, num_classes)
        )
        
        ## 원본 fc
        # self.fc = nn.Linear(out_dim, nOut)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward_once(self, x):

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.instancenorm(x).unsqueeze(1).detach()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # [bs, 128, 5, 51]
        x = torch.mean(x, dim=2, keepdim=True) # [bs,128, 1, 51]

        if self.encoder_type == "SAP":
            x = x.permute(0,3,1,2).squeeze(-1) # after permute [bs, 51, 128])
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2) # [bs, 51]
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1) # [bs, 51, 1]
            x = torch.sum(x * w, dim=1) # [bs, 128]

        elif self.encoder_type == "ASP":
            x = x.permute(0,3,1,2).squeeze(-1)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt( ( torch.sum((x**2) * w, dim=1) - mu**2 ).clamp(min=1e-5) )
            x = torch.cat((mu,rh),1)

        x = x.view(x.size()[0], -1) # [bs, 128]
        # x = self.fc(x) # [bs, out_class] if do SAP
        return x

    def forward(self, x, y):
        x = self.forward_once(x)
        y = self.forward_once(y)
        t = torch.cat([x, y], dim=-1)  # [bs, 256]

        # return self.fc(t) # num_classes 도 바꿔야함.
        t = self.fc(t).squeeze(-1) # [bs,1] -> [bs]
        return t


def ResNetSE_concat(nOut=1, **kwargs):
    # Number of filters
    num_filters = [16, 32, 64, 128]
    model = ResNetSE_C(SEBasicBlock, [3, 4, 6, 3], num_filters, nOut, **kwargs)
    return model



# https://github.com/clovaai/voxceleb_trainer/tree/a0466aa285106c631a58c0ddb8ea27805e13ef7b
class ResNetSE_ang(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP', n_mels=40, log_input=True, **kwargs):
        super(ResNetSE_ang, self).__init__()

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))
        
        self.inplanes   = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels     = n_mels
        self.log_input  = log_input

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=7, stride=(2, 1), padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(1, 1))

        self.instancenorm   = nn.InstanceNorm1d(n_mels)
        # self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=n_mels)

        if self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion
        elif self.encoder_type == "ASP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion * 2
        else:
            raise ValueError('Undefined encoder')


        self.fc = nn.Sequential(
            nn.Linear(out_dim*2, 100),  # [256] -> [100]
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(100,50),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(50,50),
            nn.GELU(),
            # nn.Linear(512, num_classes)
        )
        
        ## 원본 fc
        # self.fc = nn.Linear(out_dim, nOut)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward_once(self, x):

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.instancenorm(x).unsqueeze(1).detach()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # [bs, 128, 5, 51]
        x = torch.mean(x, dim=2, keepdim=True) # [bs,128, 1, 51]

        if self.encoder_type == "SAP":
            x = x.permute(0,3,1,2).squeeze(-1) # after permute [bs, 51, 128])
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2) # [bs, 51]
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1) # [bs, 51, 1]
            x = torch.sum(x * w, dim=1) # [bs, 128]

        elif self.encoder_type == "ASP":
            x = x.permute(0,3,1,2).squeeze(-1)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt( ( torch.sum((x**2) * w, dim=1) - mu**2 ).clamp(min=1e-5) )
            x = torch.cat((mu,rh),1)

        x = x.view(x.size()[0], -1) # [bs, 128]
        # x = self.fc(x) # [bs, out_class] if do SAP
        return x

    def forward(self, x, y):
        x = self.forward_once(x)
        y = self.forward_once(y)
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        t = torch.cat([x, y], dim=1)  # bs, 2, 256

        return t

def ResNetSE_angler(nOut=1, **kwargs):
    # Number of filters
    num_filters = [16, 32, 64, 128]
    model = ResNetSE_ang(SEBasicBlock, [3, 4, 6, 3], num_filters, nOut, **kwargs)
    return model




##################   version : 9 ##################################
class ResNetSEContra(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP', n_mels=40, log_input=True, **kwargs):
        super(ResNetSEContra, self).__init__()

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))
        
        self.inplanes   = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels     = n_mels
        self.log_input  = log_input

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=7, stride=(2, 1), padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(1, 1))

        self.instancenorm   = nn.InstanceNorm1d(n_mels)
        # self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=n_mels)

        if self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion
        elif self.encoder_type == "ASP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion * 2
        else:
            raise ValueError('Undefined encoder')


        self.fc = nn.Sequential(
            nn.Linear(out_dim*2, 100),  # [256] -> [100]
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(100,50),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(50, 1),
            # nn.Linear(512, num_classes)
        )
        
        ## 원본 fc
        # self.fc = nn.Linear(out_dim, nOut)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward_once(self, x):

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.instancenorm(x).unsqueeze(1).detach()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # [bs, 128, 5, 51]
        x = torch.mean(x, dim=2, keepdim=True) # [bs,128, 1, 51]

        if self.encoder_type == "SAP":
            x = x.permute(0,3,1,2).squeeze(-1) # after permute [bs, 51, 128])
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2) # [bs, 51]
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1) # [bs, 51, 1]
            x = torch.sum(x * w, dim=1) # [bs, 128]

        elif self.encoder_type == "ASP":
            x = x.permute(0,3,1,2).squeeze(-1)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt( ( torch.sum((x**2) * w, dim=1) - mu**2 ).clamp(min=1e-5) )
            x = torch.cat((mu,rh),1)

        x = x.view(x.size()[0], -1) # [bs, 128]
        # x = self.fc(x) # [bs, out_class] if do SAP
        return x

    def forward(self, x, y):
        x = self.forward_once(x)
        y = self.forward_once(y)
        return torch.norm(x-y, dim =-1)


def ResNetSEContrastive(nOut = 1, **kwargs):
    num_filters = [16, 32, 64, 128]
    model = ResNetSEContra(SEBasicBlock, [3, 4, 6, 3], num_filters, nOut, **kwargs)
    return model



class ResNetSE_arc(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP', n_mels=40, log_input=True, **kwargs):
        super(ResNetSE_arc, self).__init__()

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))
        
        self.inplanes   = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels     = n_mels
        self.log_input  = log_input

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=7, stride=(2, 1), padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(1, 1))

        self.instancenorm   = nn.InstanceNorm1d(n_mels)
        # self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=n_mels)

        if self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion
        elif self.encoder_type == "ASP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion * 2
        else:
            raise ValueError('Undefined encoder')


        self.fc = nn.Sequential(
            nn.Linear(out_dim*2, out_dim),  # [256] -> [100]
            nn.GELU(),
            nn.Dropout(0.3),

            # nn.Linear(out_dim*4,out_dim*2),
            # nn.GELU(),
            # nn.Dropout(0.3),

            # nn.Linear(out_dim*2, out_dim),
            # nn.Linear(512, num_classes)
        )
        
        ## 원본 fc
        # self.fc = nn.Linear(out_dim, nOut)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward_once(self, x):

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.instancenorm(x).unsqueeze(1).detach()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # [bs, 128, 5, 51]
        x = torch.mean(x, dim=2, keepdim=True) # [bs,128, 1, 51]

        if self.encoder_type == "SAP":
            x = x.permute(0,3,1,2).squeeze(-1) # after permute [bs, 51, 128])
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2) # [bs, 51]
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1) # [bs, 51, 1]
            x = torch.sum(x * w, dim=1) # [bs, 128]

        elif self.encoder_type == "ASP":
            x = x.permute(0,3,1,2).squeeze(-1)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt( ( torch.sum((x**2) * w, dim=1) - mu**2 ).clamp(min=1e-5) )
            x = torch.cat((mu,rh),1)

        x = x.view(x.size()[0], -1) # [bs, 128]
        # x = self.fc(x) # [bs, out_class] if do SAP
        return x

    def forward(self, x, y):
        x = self.forward_once(x)
        y = self.forward_once(y)
        t = torch.cat([x, y], dim=-1)  # [bs, 256]

        # return self.fc(t) # num_classes 도 바꿔야함.
        t = self.fc(t) # [bs,256] -> [bs, 128]  ###############  이걸로하니까 로스가 안줄어들어
        # t = torch.norm(x-y, dim =-1)
        r = torch.cat([t,x,y], dim = -1)
        return r


def ResNetSE_arcloss(nOut=1, **kwargs):
    # Number of filters
    num_filters = [16, 32, 64, 128]
    model = ResNetSE_arc(SEBasicBlock, [3, 4, 6, 3], num_filters, nOut, **kwargs)
    return model