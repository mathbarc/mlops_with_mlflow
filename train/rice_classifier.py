import numpy
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
import math

class RiceClassifierV1(nn.Module):

    def __init__(self, labels, conv_activation = F.leaky_relu):
        super(RiceClassifierV1,self).__init__()

        self.labels = labels
        self.nClasses = len(labels)

        self.conv_activation = conv_activation

        self.resize = torchvision.transforms.Resize((64,64))


        # 1 input image channel (grayscale), 10 output channels/feature maps
        # 3x3 square convolution kernel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 8, 3, padding="same")
        self.batch_norm1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding="same")
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, padding="same")
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3, padding="same")
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3, padding="same")
        self.batch_norm5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 256, 3, padding="same")
        self.batch_norm6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, padding="same")
        self.batch_norm7 = nn.BatchNorm2d(512)
        
        self.dropout = nn.Dropout(p=0.3)
        self.dense = nn.Linear(512, self.nClasses)
        self.output = nn.Softmax(dim=1)


    ## TODO: define the feedforward behavior
    def forward(self, input):
        x = input
        x = self.resize(x)
        if(torch.cuda.is_initialized()):
            x = x.cuda()
        # one activated conv layer
        x = self.conv_activation(self.conv1(x))
        x = self.pool(x)
        x = self.batch_norm1(x)
        x = self.conv_activation(self.conv2(x))
        x = self.pool(x)
        x = self.batch_norm2(x)
        x = self.conv_activation(self.conv3(x))
        x = self.pool(x)
        x = self.batch_norm3(x)
        x = self.conv_activation(self.conv4(x))
        x = self.pool(x)
        x = self.batch_norm4(x)
        x = self.conv_activation(self.conv5(x))
        x = self.pool(x)
        x = self.batch_norm5(x)
        x = self.conv_activation(self.conv6(x))
        x = self.pool(x)
        x = self.batch_norm6(x)
        x = self.conv_activation(self.conv7(x))
        x = self.batch_norm7(x)
        x = self.dropout(x)
        x = x.view(-1, 512)
        x = self.output(self.dense(x))

        # final output
        return x

if __name__=="__main__":
    classifier = RiceClassifierV1(["c1", "c2", "c3", "c4"])
    print(classifier)