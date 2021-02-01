import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import ConvBlock, LinearAttentionBlock, ProjectorBlock
from initialize import *

'''
attention before max-pooling
'''

class AttnVGG_before(nn.Module):
    def __init__(self, im_size, num_classes, attention=True, normalize_attn=True, init='xavierUniform', _base_features=64,dropout=0.0):
        super(AttnVGG_before, self).__init__()

        #self.base_features = 64
        self.base_features = _base_features

        self.dropout = nn.Dropout(p=dropout)

        self.attention = attention
        # conv blocks
        self.conv_block1 = ConvBlock(1, self.base_features, 2, dropout=dropout)
        self.conv_block2 = ConvBlock(self.base_features, self.base_features * 2, 2, dropout=dropout)
        self.conv_block3 = ConvBlock(self.base_features * 2, self.base_features * 4, 3, dropout=dropout)
        self.conv_block4 = ConvBlock(self.base_features * 4, self.base_features * 8, 3, dropout=dropout)
        self.conv_block5 = ConvBlock(self.base_features * 8, self.base_features * 8, 3, dropout=dropout)
        self.conv_block6 = ConvBlock(self.base_features * 8, self.base_features * 8, 2, pool=True, dropout=dropout)
        self.dense = nn.Conv2d(in_channels=self.base_features * 8, out_channels=self.base_features * 8, kernel_size=int(im_size/32), padding=0, bias=True)
        # Projectors & Compatibility functions
        if self.attention:
            self.projector = ProjectorBlock(self.base_features * 4, self.base_features * 8)
            self.attn1 = LinearAttentionBlock(in_features=self.base_features * 8, normalize_attn=normalize_attn)
            self.attn2 = LinearAttentionBlock(in_features=self.base_features * 8, normalize_attn=normalize_attn)
            self.attn3 = LinearAttentionBlock(in_features=self.base_features * 8, normalize_attn=normalize_attn)
        # final classification layer

        if self.attention:
            self.classify = nn.Linear(in_features=(self.base_features * 8)*3, out_features=num_classes, bias=True)
        else:
            self.classify = nn.Linear(in_features=self.base_features * 8, out_features=num_classes, bias=True)
        # initialize
        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self)
        else:
            raise NotImplementedError("Invalid type of initialization!")
    def forward(self, x):
        # feed forward
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        l1 = self.conv_block3(x) # /1
        x = F.max_pool2d(l1, kernel_size=2, stride=2, padding=0) # /2
        l2 = self.conv_block4(x) # /2
        x = F.max_pool2d(l2, kernel_size=2, stride=2, padding=0) # /4
        l3 = self.conv_block5(x) # /4
        x = F.max_pool2d(l3, kernel_size=2, stride=2, padding=0) # /8
        x = self.conv_block6(x) # /32
        g = self.dense(x) # batch_sizex512x1x1
        # pay attention
        if self.attention:
            c1, g1 = self.attn1(self.projector(l1), g)
            c2, g2 = self.attn2(l2, g)
            c3, g3 = self.attn3(l3, g)
            g = torch.cat((g1,g2,g3), dim=1) # batch_sizexC
            #dropout
            g = self.dropout(g)
            # classification layer
            x = self.classify(g) # batch_sizexnum_classes
        else:
            c1, c2, c3 = None, None, None
            #dropout
            g = self.dropout(g)
            # classification layer
            x = self.classify(torch.squeeze(g))
        #return [x, c1, c2, c3]

        return [torch.sigmoid(x), c1, c2, c3]
