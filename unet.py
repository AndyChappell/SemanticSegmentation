import torch.nn as nn
import torch

def maxpool():
    return nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

def dropout(prob):
    return nn.Dropout(prob)

def reinit_layer(seq_block, leak = 0.0, use_kaiming_normal=True):
    for layer in seq_block:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            print("Reinitialising", layer)
            if use_kaiming_normal:
                nn.init.kaiming_normal_(layer.weight, a = leak)
            else:
                nn.init.kaiming_uniform_(layer.weight, a = leak)
                layer.bias.data.zero_()

class InitBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(InitBlock, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(c_out)
        self.pool = nn.MaxPool2d(3, stride = 2, padding = 1)
        self.block = nn.Sequential(self.conv, self.relu, self.bn)#, self.pool)
        reinit_layer(self.block, leak = 0.0)

    def forward(self, x):
        return self.block(x)

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(ResBlock, self).__init__()
        if c_out != c_in:
            # In principle this should be stride 2 and no maxpool after
            self.conv1 = nn.Conv2d(c_in, c_out, kernel_size = 3, stride = 1, padding = 1, bias = False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size = 3, padding = 1, stride = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.block = nn.Sequential(self.conv1, self.bn1, self.relu, self.conv2, self.bn2)
        if c_out != c_in:
            # May need to become stride 2 if stride above is included again
            ds_conv = nn.Conv2d(c_in, c_out, kernel_size = 1, stride = 1, bias = False)
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

class ConvBlock(nn.Module):
    # Sigmoid activation suitable for binary cross-entropy
    def __init__(self, c_in, c_out, k_size = 3, k_pad = 1):
        # 3, 1 v 5, 2
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size = k_size, padding = k_pad, stride = 1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size = k_size, padding = k_pad, stride = 1),
            nn.BatchNorm2d(c_out))
        reinit_layer(self.block)

    def forward(self, x):
        return self.block(x)

class ConvBlockOld(nn.Module):
    # Sigmoid activation suitable for binary cross-entropy
    def __init__(self, c_in, c_out):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(c_out),
            nn.Conv2d(c_out, c_out, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(c_out))
        reinit_layer(self.block)

    def forward(self, x):
        return self.block(x)

class TransposeConvBlock(nn.Module):
    # Sigmoid activation suitable for binary cross-entropy
    def __init__(self, c_in, c_out, k_size = 3, k_pad = 1):
        # 3, 1 v 5,2
        super(TransposeConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size = k_size, padding = k_pad, output_padding = 1, stride = 2),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True))
        reinit_layer(self.block)

    def forward(self, x):
        return self.block(x)

class TransposeConvBlockOld(nn.Module):
    # Sigmoid activation suitable for binary cross-entropy
    def __init__(self, c_in, c_out):
        super(TransposeConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size = 3, padding = 1, output_padding = 1, stride = 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(c_out))
        reinit_layer(self.block)

    def forward(self, x):
        return self.block(x)

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

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        for i, module in enumerate(args):
            self.add_module(str(i), module)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class UNet(nn.Module):
    def __init__(self, in_dim, n_classes, depth = 4, n_filters = 16, drop_prob = 0.1, y_range = None):
        super(UNet, self).__init__()
        # Contracting Path
        ds_convs = []
        for i in range(depth):
            if i == 0: ds_convs.append(ConvBlock(in_dim, n_filters * 2**i))
            else: ds_convs.append(ConvBlock(n_filters * 2**(i - 1), n_filters * 2**i))
        self.ds_convs = ListModule(*ds_convs)

        ds_maxpools = []
        for i in range(depth):
            ds_maxpools.append(maxpool())
        self.ds_maxpools = ListModule(*ds_maxpools)
        
        ds_dropouts = []
        for i in range(depth):
            ds_dropouts.append(dropout(drop_prob))
        #self.ds_dropouts = ListModule(*ds_dropouts)

        self.bridge = ConvBlock(n_filters * 2**(depth - 1), n_filters * 2**depth)
        
        # Expansive Path
        us_tconvs = []
        for i in range(depth, 0, -1):
            us_tconvs.append(TransposeConvBlock(n_filters * 2**i, n_filters * 2**(i - 1)))
        self.us_tconvs = ListModule(*us_tconvs)

        us_convs = []
        for i in range(depth, 0, -1):
            us_convs.append(ConvBlock(n_filters * 2**i, n_filters * 2**(i - 1)))
        self.us_convs = ListModule(*us_convs)

        us_dropouts = []
        for i in range(depth):
            us_dropouts.append(dropout(drop_prob))
        #self.us_dropouts = ListModule(*us_dropouts)

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
        res = x
        conv_stack = []

        # Downsample
        for i in range(len(self.ds_convs)):
            res = self.ds_convs[i](res); conv_stack.append(res)
            res = self.ds_maxpools[i](res)
            #res = self.ds_dropouts[i](res)
        
        # Bridge
        res = self.bridge(res)
        
        # Upsample
        for i in range(len(self.us_convs)):
            res = self.us_tconvs[i](res)
            res = torch.cat([res, conv_stack.pop()], dim=1)
            #res = self.us_dropouts[i](res)
            res = self.us_convs[i](res)

        output = self.output(res)
        #print(output.size())
        return output
