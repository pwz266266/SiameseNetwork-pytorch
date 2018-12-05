#This file is expected to be a prototype of convolutional network

import torch
import torch.nn as nn
import torch.nn.functional as F

class Maxout(nn.Module):

    def __init__(self, in_size, out_size, pool_size):
        super().__init__()
        self.in_size, self.out_size, self.pool_size = in_size, out_size, pool_size
        self.lin = nn.Linear(in_size, out_size * pool_size)


    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.out_size
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        #Explanation of Conv2d(x,y,z):
        #      Build a convolutional layer with
        #            x input channels
        #            y output channels
        #            z*z filter
        self.model1 = nn.Sequential(
        nn.Conv2d(3,64,7,stride=2,padding=3),
        nn.MaxPool2d(3,stride=2,padding=1)
        )

        self.model2 = nn.Sequential(
        nn.Conv2d(64,64,1,stride=1),
        nn.Conv2d(64,192,3,stride=1,padding=1)
        )

        self.model3 = nn.Sequential(
        nn.MaxPool2d(3,stride=2,padding=1),
        nn.Conv2d(192,192,1,stride=1),
        nn.Conv2d(192,384,3,stride=1,padding=1),
        nn.MaxPool2d(3,stride=2,padding=1),
        nn.Conv2d(384,384,1,stride=1),
        nn.Conv2d(384,256,3,stride=1,padding=1),
        nn.Conv2d(256,256,1,stride=1),
        nn.Conv2d(256,256,3,stride=1,padding=1),
        nn.Conv2d(256,256,1,stride=1),
        nn.Conv2d(256,256,3,stride=1,padding=1),
        nn.MaxPool2d(3,stride=2,padding=1)
        )

        self.model4 = nn.Sequential(
        Maxout(7*7*256, 1*32*128, 2),
        Maxout(1*32*128, 1*32*128, 2),
        nn.Linear(1*32*128, 1*1*128),
        nn.Linear(128,128)
        )


    #Fucntion to feed forward

    def forward(self, x): #Go through network once
        # Max pooling over a (2, 2) window
        x = x.type('torch.cuda.FloatTensor')
        x = F.normalize(x)
        x = self.model1(x)
        x = F.normalize(x)
        x = self.model2(x)
        x = F.normalize(x)
        x = self.model3(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.model4(x)

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
