# Code by GunhoChoi
import torchvision

from FusionNet import *
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
imageSize = 1024
learningRate = 0.0002
nEpochs = 2

# Input folder structure root/class1/Img1.jpg must be adhered
# Input training data
inputDirectoryTrain = "./TestData/train"
imageDataTrain = datasets.ImageFolder(root=inputDirectoryTrain, transform = transforms.Compose([transforms.Resize(size=imageSize), transforms.CenterCrop(size=(imageSize,imageSize*2)), transforms.ToTensor(),]))
imageBatchTrain = data.DataLoader(imageDataTrain, batch_size=batch_size, shuffle=True, num_workers=2)

# Input validation data
inputDirectoryValidation = "./TestData/val/"
imageDataValidation = datasets.ImageFolder(root=inputDirectoryValidation, transform = transforms.Compose([transforms.Resize(size=imageSize), transforms.CenterCrop(size=(imageSize,imageSize*2)), transforms.ToTensor(),]))
imageBatchValidation = data.DataLoader(imageDataValidation, batch_size=batch_size, shuffle=True, num_workers=2)

nFiltersInitial = 16 # Initially 64

# Initiate Generator
if args.network == "fusionnet":
    generator = nn.DataParallel(FusionGenerator(3,3,nFiltersInitial),device_ids=[i for i in range(args.num_gpu)]) #.cuda()
elif args.network == "unet":
    generator = nn.DataParallel(UnetGenerator(3,3,nFiltersInitial,True),device_ids=[i for i in range(args.num_gpu)]) #.cuda()

# Load pretrained model if it exists
try:
    generator = torch.load('./model/{}.pkl'.format(args.network))
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass

# Define the loss function and optimizer
loss_fn = nn.MSELoss()
gen_optimizer = torch.optim.Adam(generator.parameters(),lr=learningRate)

# Training
lossTrainAverageList = []
lossValidationAverageList = []
epochList = []

for epoch in range(nEpochs):
    epochList.append(epoch)

    lossTrainAverage = 0
    trainSampleCounter = 0

    # Data Loader allows us to process the training and validation samples in batches, rather than all at once,
    # which is essential for large data sets
    for index, (image,label) in enumerate(imageBatchTrain):
        trainSampleCounter += 1
        inputImage, truthImage = torch.chunk(image, chunks=2, dim=3)
        gen_optimizer.zero_grad()

        x = Variable(inputImage) #.cuda(0)
        y_truth = Variable(truthImage) #.cuda(0)

        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.
        y_pred = generator.forward(x)

        sys.exit()

        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the loss.
        loss = loss_fn(y_pred, y_truth)
        lossTrainAverage += loss.item()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # Optimise the weights
        gen_optimizer.step()

        if index == 0:
            traced_script_module = torch.jit.trace(generator.module, x)
            print("saving module")
            traced_script_module.save("model.pt")

        if index % 400 ==0:
            print(epoch)
            print(loss)
            v_utils.save_image(x.cpu().data,"./result/InputImage_Epoch_{}_FileIndex_{}.png".format(epoch,index))
            v_utils.save_image(y_truth.cpu().data,"./result/TruthImage_Epoch_{}_FileIndex_{}.png".format(epoch,index))
            v_utils.save_image(y_pred.cpu().data,"./result/GeneratedImage_Epoch_{}_FileIndex_{}.png".format(epoch,index))
            torch.save(generator,'./model/{}.pkl'.format(args.network))

    lossTrainAverage /= trainSampleCounter
    lossTrainAverageList.append(lossTrainAverage)

    lossValidationAverage = 0
    validationSampleCounter = 0

    for index, (image,label) in enumerate(imageBatchValidation):
        validationSampleCounter += 1
        inputImage, truthImage = torch.chunk(image, chunks=2, dim=3)

        x = Variable(inputImage) #.cuda(0)
        y_truth = Variable(truthImage) #.cuda(0)
        y_pred = generator.forward(x)

        loss = loss_fn(y_pred, y_truth)
        lossValidationAverage += loss.item()

    lossValidationAverage /= validationSampleCounter
    lossValidationAverageList.append(lossValidationAverage)

fig, ax = plt.subplots()
ax.set_title('Model Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Average MSE Loss')
ax.plot(epochList, lossTrainAverageList, label='Training Sample', c='darkred')
ax.plot(epochList, lossValidationAverageList, label='Validation Sample', c='forestgreen')
ax.legend(loc='upper right')
plt.savefig('AverageLossVsTrainingEpoch')

with open('LossData.txt', 'w') as dataFile:
    for idx, item in enumerate(epochList):
        dataFile.write('Epoch : ' + str(item) + ', Train Loss : ' + str(lossTrainAverageList[idx]) + ', Validation Loss : ' + str(lossValidationAverageList[idx]) + '\n')

