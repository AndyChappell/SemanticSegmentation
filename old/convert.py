# Code by GunhoChoi
import torchvision

from UNet import *
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--network",type=str,default="unet",help="choose between fusionnet & unet")
parser.add_argument("--batch_size",type=int,default=1,help="batch size")
parser.add_argument("--num_gpu",type=int,default=3,help="number of gpus")
args = parser.parse_args()

# hyperparameters

batch_size = args.batch_size
imageSize = 512
learningRate = 0.0002
epoch = 50

#torch.nn.Module.dump_patches = True

# input pipeline

inputDirectory = "./Pandora_ExampleData/"
imageData = datasets.ImageFolder(root=inputDirectory, transform = transforms.Compose([transforms.Resize(size=imageSize), transforms.CenterCrop(size=(imageSize,imageSize*2)), transforms.ToTensor(),]))
imageBatch = data.DataLoader(imageData, batch_size=batch_size, shuffle=True, num_workers=2)

# initiate Generator
#generator = nn.DataParallel(UnetGenerator(3,3,16),device_ids=[i for i in range(args.num_gpu)])

# load pretrained model

print("Loading model")
#try:
#    device = torch.device('cpu')
#    print('1')
#    generator = TheModelClass(*args, **kwargs)
#    print('2')
generator = torch.load('model/unet.pkl',  map_location='cpu') #lambda storage, loc: storage) # map_location=device))

#    print('3')
print("\n--------model restored--------\n")
#except:
#    print("\n--------model not restored--------\n")
#    pass

# loss function & optimizer

recon_loss_func = nn.MSELoss()
gen_optimizer = torch.optim.Adam(generator.parameters(),lr=learningRate)

# training

file = open('./{}_mse_loss'.format(args.network), 'w')
for i in range(epoch):
    for _,(image,label) in enumerate(imageBatch):
        satel_image, map_image = torch.chunk(image, chunks=2, dim=3) 
        print("In loop")
        gen_optimizer.zero_grad()

        x = Variable(satel_image)
        print(x)
        print(x.size())
        y = generator.forward(x)
        print(y)
        print(y.size())
#        sys.exit()
#        y_ = Variable(map_image)
#        y = generator.forward(x)
#        loss = recon_loss_func(y,y_)
#        file.write(str(loss)+"\n")
#        loss.backward()
#        gen_optimizer.step()

        traced_script_module = torch.jit.trace(generator.module, x)
        print("saving module")
        traced_script_module.save("model_cpu.pt")
        sys.exit()
