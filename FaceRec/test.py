import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import *
from dataset import *
from train import *

def runTest():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    composed = transforms.Compose([Rescale(IMAGE_SIZE),ToTensor()])
    network = SiameseNetwork()
    network.load_state_dict(torch.load("./"+MODELNAME))
    network.eval()
    network.to(device)
    labelSet = {}

    #Give each label a standard output
    trainedDataset = FaceDataset(csv_file = TEST_FILE, root_dir = IMAGE_DIRECTORY, forTrain = False, transform = composed)
    trainedDataloader = DataLoader(trainedDataset, batch_size=1, shuffle=False, num_workers=10)
    for i, data in enumerate(trainedDataloader, 0):
        sample, label = data
        if i%1000 == 0:
            print("Go through"+ str(i) +  "images")
        sample.to(device)
        labelSet[label] = network(sample).data
    print('Standard output setting: Done.')

    #Go through all trainning data and test the accuracy
    testDataset = FaceDataset(csv_file = TEST_FILE, root_dir = IMAGE_DIRECTORY, forTrain = False, transform = composed)
    testDataloader = DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=10)

    total_count = 0
    correct_count = 0
    for i, data in enumerate(trainedDataloader, 0):
        sample, label = data
        sample.to(device)
        output = network(sample).data
        distance = 100000
        curr_label = '0'
        for q in labelSet:
            currDis = (output - labelSet[q]).pow(2).sum(1)
            if currDis<distance:
                distance = currDis
                curr_label = q
        total_count += 1 
        if label == curr_label:
            correct_count += 1
        if total_count % 100 == 0:
            print('The total correctness rate = %.1f%%' %(correct_count/total_count*100))

    correct_count -= len(labelSet)
    total_count -= len(labelSet)
    print('The total correctness rate = %.1f%%' %(correct_count/total_count*100))
