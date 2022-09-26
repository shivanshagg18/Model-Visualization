import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return x * F.sigmoid(x)
    
class ResNetBlock(nn.Module):
    def __init__(self, in_depth, hidden_depth=None, out_depth=None, stride=1, dilation=1,
                 batchnorm=True, activation='swish', zero_output=True, bottleneck=True):
        super(ResNetBlock, self).__init__()
        if out_depth is None:
            out_depth = in_depth * stride
        if stride > 1:
            self.shortcut_layer = nn.Conv3d(in_depth, out_depth, kernel_size=3, stride=stride,
                                            padding=1, dilation=dilation, bias=True)
        else:
            self.shortcut_layer = None

        layers = []
        if bottleneck:
            if hidden_depth is None:
                hidden_depth = in_depth // 4
            k_sizes = [3, 1, 3]
            depths = [in_depth, hidden_depth, hidden_depth, out_depth]
            paddings = [1, 0, 1]
            strides = [1, 1, stride]
            dilations = [dilation, 1, dilation]
        else:
            if hidden_depth is None:
                hidden_depth = in_depth
            k_sizes = [3, 3]
            depths = [in_depth, hidden_depth, out_depth]
            paddings = [1, 1]
            strides = [1, stride]
            dilations = [dilation, dilation]
        
        for i in range(len(k_sizes)):
            if batchnorm:
                layers.append(nn.BatchNorm3d(depths[i], eps=1e-8))
            layers.append(Swish())
            layers.append(nn.Conv3d(depths[i], depths[i + 1], k_sizes[i], padding=paddings[i],
                                    stride=strides[i], dilation=dilations[i], bias=False))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        Fx = self.layers(x)
        if self.shortcut_layer is not None:
            x = self.shortcut_layer(x)
        return x + Fx
    
class MLP(nn.Module):
    def __init__(self, in_depth, hidden_depths, out_depth, activation='swish', batchnorm=True, dropout=0.):
        super(MLP, self).__init__()
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.depths = [in_depth, *hidden_depths, out_depth]

        self.linear_layers = nn.ModuleList([])
        self.norm = nn.ModuleList([])
        self.act = nn.ModuleList([])

        for i in range(len(self.depths) - 1):
            self.linear_layers.append(nn.Linear(self.depths[i], self.depths[i + 1], bias=not batchnorm))
            if i != len(self.depths) - 2:
                if batchnorm:
                    self.norm.append(nn.BatchNorm1d(self.depths[i + 1], eps=1e-8))
                self.act.append(Swish())

    def forward(self, x):
        for i in range(len(self.depths) - 1):
            if self.dropout > 0.:
                x = F.dropout(x, self.dropout, self.training)
            x = self.linear_layers[i](x)
            if i != len(self.depths) - 2:
                if self.batchnorm:
                    x = self.norm[i](x)
                x = self.act[i](x)
        return x
    
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.init_conv = nn.Conv3d(1, 8, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3))
        self.init_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.seq1 = nn.Sequential(
            ResNetBlock(8),
            ResNetBlock(8)
        )
        self.seq2 = nn.Sequential(
            ResNetBlock(8, out_depth=8, stride=2),
            ResNetBlock(8)
        )
        self.seq3 = nn.Sequential(
            ResNetBlock(8, out_depth=16, stride=2),
            ResNetBlock(16)
        )
        self.seq4 = nn.Sequential(
            ResNetBlock(16, out_depth=32, stride=2),
            ResNetBlock(32)
        )
        self.seq5 = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            MLP(32, [32*2], 32, dropout=0.5),
            nn.Tanh(),
            MLP(32, [32*2], 2, dropout=0.5),
            nn.LogSoftmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(6*7*6*32, 128),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(64, 2),
            nn.LogSoftmax(dim=1))
        self.last_layer_output = None
        
    def forward(self, x):
        out = self.init_conv(x)
        out = self.init_pool(out)
        out = self.seq1(out)
#         out = self.seq2(out)
        out = self.seq3(out)
        out = self.seq4(out)
#         out = self.seq5(out)
        self.last_layer_output = out
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out