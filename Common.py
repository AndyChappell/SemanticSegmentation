import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

def conv_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

#========================================================================

def conv_trans_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

#========================================================================

def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool

#========================================================================

def conv_block_2(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model

#========================================================================

def conv_block_3(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        conv_block(out_dim,out_dim,act_fn),
        nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model

#========================================================================

def Accuracy(inputTensor, truthTensor):
    if (inputTensor.size()[0] is not 1) or (inputTensor.size()[1] is not 3) or (truthTensor.size()[0] is not 1) or (truthTensor.size()[1] is not 3):
        print('Tensors have different batch and/or channel sizes')
        return 0

    if (inputTensor.size()[2] != truthTensor.size()[2]) or (inputTensor.size()[3] != truthTensor.size()[3]):
        print('Tensors have different height and/or width sizes')
        return 0

    nPixels = 0
    nCorrectPixels = 0

    for height in range(inputTensor.size()[2]):
        for depth in range(inputTensor.size()[3]):
            redTruth = truthTensor[0][0][height][depth].item()
            blueTruth = truthTensor[0][2][height][depth].item()

            if (redTruth > 0) or (blueTruth > 0):
                nPixels += 1
                redPred = inputTensor[0][0][height][depth].item()
                bluePred = inputTensor[0][2][height][depth].item()

                if (redTruth > blueTruth and redPred > bluePred) or (blueTruth > redTruth and bluePred > redPred):
                    nCorrectPixels += 1

    return nCorrectPixels/nPixels
