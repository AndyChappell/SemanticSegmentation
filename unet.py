import torch.nn as nn
import torch

import torch.nn as nn
import torch

def get_stats(x):
    return x.mean(), x.std()

def reinit_layer(seq_block, leak = 0.0, use_kaiming_normal=True):
    for layer in seq_block:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            if use_kaiming_normal:
                nn.init.kaiming_normal_(layer.weight, a = leak)
            else:
                nn.init.kaiming_uniform_(layer.weight, a = leak)
                layer.bias.data.zero_()

def conv2d_block(c_in, c_out):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    block = nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size = 3, padding = 1, stride = 1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.Conv2d(c_out, c_out, kernel_size = 3, padding = 1, stride = 1),
        nn.BatchNorm2d(c_out),
        nn.ReLU())
    reinit_layer(block, leak = 0.0)
    
    return block

def maxpool():
    return nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

def conv2d_block_transpose(n_fil_in, n_fil_out):
    block = nn.Sequential(
        nn.ConvTranspose2d(n_fil_in, n_fil_out, kernel_size = 3, padding = 1, output_padding = 1, stride = 2),
        nn.BatchNorm2d(n_fil_out),
        nn.ReLU())
    reinit_layer(block, leak = 0.0)
    
    return block

def dropout(prob):
    return nn.Dropout(prob)

class Sigmoid(nn.Module):
    # Sigmoid activation suitable for binary cross-entropy
    def __init__(self):
        super(Sigmoid, self).__init__()
    
    def forward(self, x):
        return torch.sigmoid(x)

class SigmoidRange(nn.Module):
    # Sigmoid activation suitable for categorical cross-entropy
    def __init__(self, low, high):
        super(SigmoidRange, self).__init__()
        self.low = low
        self.high = high
    
    def forward(self, x):
        return torch.sigmoid(x) * (self.high - self.low) + self.low

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size = 3, padding = 1, stride = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.block = nn.Sequential(self.conv1, self.bn1, self.relu, self.conv2, self.bn2)
        if c_out != c_in:
            ds_conv = nn.Conv2d(c_in, c_out, kernel_size = 3, stride = 1, padding = 1, bias = False)
            ds_bn = nn.BatchNorm2d(c_out)
            self.downsample = nn.Sequential(ds_conv, ds_bn)
        else:
            self.downsample = None
        reinit_layer(self.block, leak = 0.0)

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.downsample: identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class UNet(nn.Module):
    def __init__(self, in_dim, n_classes, n_filters = 16, drop_prob = 0.1, y_range = None):
        super(UNet, self).__init__()
        # Contracting Path
        self.c1 = ResBlock(in_dim, n_filters * 1)
        self.m1 = maxpool()
        self.d1 = dropout(drop_prob)

        self.c2 = ResBlock(n_filters * 1, n_filters * 2)
        self.m2 = maxpool()
        self.d2 = dropout(drop_prob)

        self.c3 = ResBlock(n_filters * 2, n_filters * 4)
        self.m3 = maxpool()
        self.d3 = dropout(drop_prob)
        
        self.c4 = ResBlock(n_filters * 4, n_filters * 8)
        self.m4 = maxpool()
        self.d4 = dropout(drop_prob)
        
        self.c5 = ResBlock(n_filters * 8, n_filters * 16)
        
        # Expansive Path
        self.t6 = conv2d_block_transpose(n_filters * 16, n_filters * 8)
        self.d6 = dropout(drop_prob)
        self.c6 = ResBlock(n_filters * 16, n_filters * 8)

        self.t7 = conv2d_block_transpose(n_filters * 8, n_filters * 4)
        self.d7 = dropout(drop_prob)
        self.c7 = ResBlock(n_filters * 8, n_filters * 4)

        self.t8 = conv2d_block_transpose(n_filters * 4, n_filters * 2)
        self.d8 = dropout(drop_prob)
        self.c8 = ResBlock(n_filters * 4, n_filters * 2)
        
        self.t9 = conv2d_block_transpose(n_filters * 2, n_filters * 1)
        self.d9 = dropout(drop_prob)
        self.c9 = ResBlock(n_filters * 2, n_filters * 1)
        
        # Assume 3 classes to figure out
        if y_range is not None:
            self.output = nn.Sequential(
                nn.Conv2d(n_filters * 1, n_classes, 1),
                #Sigmoid(),
                SigmoidRange(*y_range)
            )
        else:
            self.output = nn.Sequential(
                nn.Conv2d(n_filters * 1, n_classes, 1)
            )

    def forward(self, x):
        # Downsample
        identity = x
        c1 = self.c1(x)
        m1 = self.m1(c1)
        d1 = self.d1(m1)
        
        c2 = self.c2(d1)
        m2 = self.m2(c2)
        d2 = self.d2(m2)

        c3 = self.c3(d2)
        m3 = self.m3(c3)
        d3 = self.d3(m3)

        c4 = self.c4(d3)
        m4 = self.m4(c4)
        d4 = self.d4(m4)

        # Bridge
        bridge = self.c5(d4)
        
        # Upsample
        
        t6 = self.t6(bridge)
        concat6 = torch.cat([t6, c4], dim=1)
        d6 = self.d6(concat6)
        c6 = self.c6(d6)

        t7 = self.t7(c6)
        concat7 = torch.cat([t7, c3], dim=1)
        d7 = self.d7(concat7)
        c7 = self.c7(d7)

        t8 = self.t8(c7)
        concat8 = torch.cat([t8, c2], dim=1)
        d8 = self.d8(concat8)
        c8 = self.c8(d8)

        t9 = self.t9(c8)
        concat9 = torch.cat([t9, c1], dim=1)
        d9 = self.d9(concat9)
        c9 = self.c9(d9)

        # Output
        output = self.output(c9)
        #print(output.size())
        return output
