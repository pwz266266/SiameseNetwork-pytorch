import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import *
from dataset import *
import sys


#Configuration
IMAGE_DIRECTORY = '/data/psywp1/img_align_celeba/'
IMAGE_SIZE = [220, 220]
TRAIN_FILE = '~/TrainSet.txt'
#VALIDATION_FILE = 'C:/Users/zy221/Desktop/Anno/ValidSet.txt'
TEST_FILE = '~/TestSet.txt'
BATCH_SIZE = 130
MODELNAME = "trainedmodel"+".pt"

class TripletLoss(nn.Module):
    def __init__(self, margin = 0.2):
        super(TripletLoss,self).__init__()
        self.margin= margin

    def forward(self, anchor, positive, negative):
        APDistance = (anchor - positive).pow(2).sum(1)
        ANDistance = (anchor - negative).pow(2).sum(1)
        loss = F.relu(APDistance - ANDistance + self.margin)
        return loss.sum()

def runTrain():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    composed = transforms.Compose([Rescale(IMAGE_SIZE),ToTensor()])
    trainDataset = FaceDataset(csv_file = TRAIN_FILE, root_dir = IMAGE_DIRECTORY, forTrain = True, transform = composed)
    trainDataloader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
    net = SiameseNetwork()
    net = net.to(device)
    criterion = TripletLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.05)
    for epoch in range(24):
        for i, data in enumerate(trainDataloader,0):
            anchor, positive, negative = data
            anchor.to(device)
            positive.to(device)
            negative.to(device)
            anchor_out, positive_out, negative_out = net.forward_triple(anchor, positive, negative)
            optimizer.zero_grad()
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('Current loss: %.3f' %(float(loss.data.cpu().numpy())))
        print('Whole dataset trained time: '+str(epoch+1))
    print('Training done.')
    torch.save(net.state_dict(), "./"+MODELNAME)
