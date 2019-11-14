from Common import *
import numpy as np

class UnetGenerator(nn.Module):

    def __init__(self,in_dim,out_dim,num_filter,plot=False):
        super(UnetGenerator,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        self.plot = plot
        act_fn = nn.LeakyReLU(0.2, inplace=True)

        print("\n------Initiating U-Net------\n")

        self.down_1 = conv_block_2(self.in_dim,self.num_filter,act_fn)
        self.pool_1 = maxpool()
        self.down_2 = conv_block_2(self.num_filter*1,self.num_filter*2,act_fn)
        self.pool_2 = maxpool()
        self.down_3 = conv_block_2(self.num_filter*2,self.num_filter*4,act_fn)
        self.pool_3 = maxpool()
        self.down_4 = conv_block_2(self.num_filter*4,self.num_filter*8,act_fn)
        self.pool_4 = maxpool()

        self.bridge = conv_block_2(self.num_filter*8,self.num_filter*16,act_fn)

        self.trans_1 = conv_trans_block(self.num_filter*16,self.num_filter*8,act_fn)
        self.up_1 = conv_block_2(self.num_filter*16,self.num_filter*8,act_fn)
        self.trans_2 = conv_trans_block(self.num_filter*8,self.num_filter*4,act_fn)
        self.up_2 = conv_block_2(self.num_filter*8,self.num_filter*4,act_fn)
        self.trans_3 = conv_trans_block(self.num_filter*4,self.num_filter*2,act_fn)
        self.up_3 = conv_block_2(self.num_filter*4,self.num_filter*2,act_fn)
        self.trans_4 = conv_trans_block(self.num_filter*2,self.num_filter*1,act_fn)
        self.up_4 = conv_block_2(self.num_filter*2,self.num_filter*1,act_fn)

        self.out = nn.Sequential(
            nn.Conv2d(self.num_filter,self.out_dim,3,1,1),
            nn.Tanh(),
        )

    def forward(self,input):
        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)

        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)

        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1,down_4],dim=1)
        up_1 = self.up_1(concat_1)

        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2,down_3],dim=1)
        up_2 = self.up_2(concat_2)

        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3,down_2],dim=1)
        up_3 = self.up_3(concat_3)

        trans_4 = self.trans_4(up_3)
        concat_4 = torch.cat([trans_4,down_1],dim=1)
        up_4 = self.up_4(concat_4)

        out = self.out(up_4)

        if self.plot:
            print('Input Size : ' + str(input))
            self.DrawTensor(input, 'Input')

            print('Down 1 Size : ' + str(down_1.size()))
            self.DrawTensor(down_1, 'Down1')
            print('Pool 1 Size : ' + str(pool_1.size()))
            self.DrawTensor(pool_1, 'Pool1')

            print('Down 2 Size : ' + str(down_2.size()))
            self.DrawTensor(down_2, 'Down2')
            print('Pool 2 Size : ' + str(pool_2.size()))
            self.DrawTensor(pool_2, 'Pool2')

            print('Down 3 Size : ' + str(down_3.size()))
            self.DrawTensor(down_3, 'Down3')
            print('Pool 1 Size : ' + str(pool_3.size()))
            self.DrawTensor(pool_3, 'Pool3')

            print('Down 4 Size : ' + str(down_4.size()))
            self.DrawTensor(down_4, 'Down4')
            print('Pool 4 Size : ' + str(pool_4.size()))
            self.DrawTensor(pool_4, 'Pool4')

            print('Bridge Size : ' + str(bridge.size()))
            self.DrawTensor(bridge, 'Bridge')

            print('Transpose 1 Size : ' + str(trans_1.size()))
            self.DrawTensor(trans_1, 'Transpose1')
            print('Concat 1 Size : ' + str(concat_1.size()))
            # No need to draw concatenation as it is merging two already saved sets of images
            print('UpSample 1 Size : ' + str(up_1.size()))
            self.DrawTensor(up_1, 'UpSample1')

            print('Transpose 2 Size : ' + str(trans_2.size()))
            self.DrawTensor(trans_2, 'Transpose2')
            print('Concat 2 Size : ' + str(concat_2.size()))
            print('UpSample 2 Size : ' + str(up_2.size()))
            self.DrawTensor(up_2, 'UpSample2')

            print('Transpose 3 Size : ' + str(trans_3.size()))
            self.DrawTensor(trans_3, 'Transpose3')
            print('Concat 3 Size : ' + str(concat_3.size()))
            print('UpSample 3 Size : ' + str(up_3.size()))
            self.DrawTensor(up_3, 'UpSample3')

            print('Transpose 4 Size : ' + str(trans_4.size()))
            self.DrawTensor(trans_4, 'Transpose4')
            print('Concat 4 Size : ' + str(concat_4.size()))
            print('UpSample 4 Size : ' + str(up_4.size()))
            self.DrawTensor(up_4, 'UpSample4')
            print('Output : ' + str(out.size()))
            self.DrawTensor(out, 'Output')

        return out

    def DrawTensor(self, inputTensor, tensorName = 'DefaultName'):
        # Tensors in PyTorch are typically 4 dimensional.  Typically : Batch Size(1) x Channels x Height x Width
        # Goal here is to get pictures of Height x Width
        tensor = torch.zeros([1, inputTensor.size()[2], inputTensor.size()[3]])

        # Check tensor shape is as expected
        if (inputTensor.size()[0] is not 1):
            print('Unexpected tensor shape, exiting...')
            sys.exit()

        # Loop over all available channels
        for channelId in range(inputTensor.size()[1]):
            tensor[0] = inputTensor[0][channelId]
            img = transforms.ToPILImage()(tensor)
            imageName = tensorName + '_ChannelId' + str(channelId) + '.png'
            img.save(imageName)
