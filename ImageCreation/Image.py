import os
import cv2
import csv
import sys
import matplotlib.pyplot as plt
import re
import numpy as np
import glob
from PIL import Image

#========================================================================================

def MakeImages(inputFile, outputFolder = '', imageSize = 1024):
    f = open(inputFile, 'r')
    reader = csv.reader(f)
    name = os.path.splitext(os.path.basename(inputFile))[0]

    isU = True if "CaloHitListU" in inputFile else False
    isV = True if "CaloHitListV" in inputFile else False
    isW = True if "CaloHitListW" in inputFile else False

    span = 980
    xLow = -420
    xHigh = xLow + span
    zLow = -350 if isU else 0 if isV else -25
    zHigh = zLow + span

    xBinEdges = np.linspace(xLow, xHigh, imageSize+1)
    zBinEdges = np.linspace(zLow, zHigh, imageSize+1)

    for row in reader:
        ProcessPicture(row, xBinEdges, zBinEdges, imageSize, name, outputFolder)

    f.close()

#========================================================================================

def ProcessPicture(row, xBinEdges, zBinEdges, imageSize, name, outputFolder):
    x = []
    z = []
    r = []
    g = []
    b = []
    row.pop(0) # Date Time
    row.pop(0) # NHits
    row.pop() # 1

    nElements = 5

    # Check the row contains the correct number of entriess
    if len(row) % nElements != 0:
        print('Missing information in input file')

    showerPdgCodes = [11, -11, 22]

    for hitIndex in range(int(len(row) / nElements)):
        # Skip if hit originates from non standard particle
        if 'e' in row[nElements * hitIndex + 3]:
            continue

        xPositionIndex = nElements * hitIndex + 0
        zPositionIndex = nElements * hitIndex + 2
        pdgIndex = nElements * hitIndex + 3
        nuanceCodeIndex = nElements * hitIndex + 4

        x.append(float(row[xPositionIndex]))
        z.append(float(row[zPositionIndex]))
        pdg = int(row[pdgIndex])
        nuanceCode = int(row[nuanceCodeIndex])

        if nuanceCode == 3000:
            r.append(0)
            g.append(0)
            b.append(1)
        elif nuanceCode == 2000:
            r.append(1)
            g.append(0)
            b.append(0)
        elif nuanceCode == 2001:
            r.append(0)
            g.append(1)
            b.append(0)

    x = np.array(x)
    z = np.array(z)

    xBinIndices = np.digitize(x, xBinEdges)
    zBinIndices = np.digitize(z, zBinEdges)

    # Build input histogram
    inputHistogram, xBinEdges, zBinEdges = np.histogram2d(x, z, bins = (xBinEdges, zBinEdges))
    inputHistogram = inputHistogram * float(255)

    inputImageName = os.path.join(outputFolder, "InputImage_" + name + "_0.jpg")
    cv2.imwrite(inputImageName, inputHistogram)

#    for rotation in [90, 180, 270]:
#        originalImage = Image.open(inputImageName)
#        newImage = originalImage.rotate(rotation)
#        newImageName = os.path.join(outputFolder, "InputImage_" + name + "_" + str(int(rotation)) + ".jpg")
#        newImage.save(newImageName)

    # Build input histogram
    truthHistogram = np.zeros((imageSize,imageSize,3), 'uint8')

    for idx, xIter in enumerate(x):
        indexX = xBinIndices[idx]
        indexZ = zBinIndices[idx]
        if indexX < imageSize and indexZ < imageSize:
            truthHistogram[indexX, indexZ] = [r[idx]*255, g[idx]*255, b[idx]*255]

    truthImageName = os.path.join(outputFolder, "TruthImage_" + name + "_0.jpg")
    cv2.imwrite(truthImageName, truthHistogram)

#    for rotation in [90, 180, 270]:
#        originalImage = Image.open(truthImageName)
#        newImage = originalImage.rotate(rotation)
#        newImageName = os.path.join(outputFolder, "TruthImage_" + name + "_" + str(int(rotation)) + ".jpg")
#        newImage.save(newImageName)

#========================================================================================

for momentum in [1]: #3,5,7
    fileDirectory = '/r07/dune/sg568/LAr/Jobs/protoDUNE/2019/September/ProtoDUNE_HierarchyMetrics_DeepLearning_Training/AnalysisTag3/mcc11_Pndr/Beam_Cosmics/' + str(momentum) + 'GeV/NoSpaceCharge/TxtFiles'
    txtFormat = 'ProtoDUNE_HierarchyMetrics_DeepLearning_Training_Job_Number_*_CaloHitList*.txt'
    outputFolder = '/r07/dune/sg568/LAr/Jobs/protoDUNE/2019/September/ProtoDUNE_HierarchyMetrics_DeepLearning_Training/AnalysisTag3/mcc11_Pndr/Beam_Cosmics/' + str(momentum) + 'GeV/NoSpaceCharge/Images'

    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    allFiles = [f for f in glob.glob(os.path.join(fileDirectory, txtFormat))] #, recursive=True)]
    allFiles.sort()

    for fileName in allFiles:
        print(fileName)
        MakeImages(fileName, outputFolder)
