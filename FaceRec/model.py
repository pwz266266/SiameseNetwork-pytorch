#This file is expected to be a prototype of convolutional network

import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        #Explanation of Conv2d(x,y,z):
        #      Build a convolutional layer with
        #            x input channels
        #            y output channels
        #            z*z filter
        self.conv1 = nn.Conv2d(3,4,2)
        self.conv2 = nn.Conv2d(4,6,2)
        self.conv3 = nn.Conv2d(6,9,2)
        self.conv4 = nn.Conv2d(9,12,3)
        self.conv5 = nn.Conv2d(12,14,3)

        # Linear(x,y) function to build fully connected layer.
        self.fc1 = nn.Linear(3276, 1240)
        self.fc2 = nn.Linear(1240, 640)
        self.fc3 = nn.Linear(640, 100)

    #Fucntion to feed forward

    def forward(self, x): #Go through network once
        # Max pooling over a (2, 2) window
        x = x.type('torch.cuda.FloatTensor')
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = F.max_pool2d(F.relu(self.conv5(x)), 2)


        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_triple(self, anchor, positive, negative): #Use for computing triplet loss
        anchor_output = self.forward(anchor)
        positive_output = self.forward(positive)
        negative_output = self.forward(negative)
        return anchor_output, positive_output, negative_output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
