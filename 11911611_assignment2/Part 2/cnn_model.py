from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):

  def __init__(self, n_channels, n_classes):
    """
    Initializes CNN object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
    """
    super(CNN, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes
    # Convolution layers
    self.conv1 = nn.Conv2d(n_channels, 64, 3, padding=1)
    self.conv1_bn = nn.BatchNorm2d(64)
    self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
    self.bn2 = nn.BatchNorm2d(128)
    self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
    self.bn3 = nn.BatchNorm2d(256)
    self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
    self.bn4 = nn.BatchNorm2d(256)
    self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
    self.bn5 = nn.BatchNorm2d(512)
    self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
    self.bn6 = nn.BatchNorm2d(512)
    self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
    self.bn7 = nn.BatchNorm2d(512)
    self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
    self.bn8 = nn.BatchNorm2d(512)
    # Pooling layers
    self.pool = nn.MaxPool2d(3, 2, 1)
    # Fully connected
    self.fc1 = nn.Linear(512 , 10)

  def forward(self, x):
    """
    Performs forward pass of the input.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """
    x = F.relu(self.conv1_bn(self.conv1(x)))
    x = self.pool(x)
    x = F.relu(self.bn2(self.conv2(x)))
    x = self.pool(x)
    x = F.relu(self.bn3(self.conv3(x)))
    x = F.relu(self.bn4(self.conv4(x)))
    x = self.pool(x)
    x = F.relu(self.bn5(self.conv5(x)))
    x = F.relu(self.bn6(self.conv6(x)))
    x = self.pool(x)
    x = F.relu(self.bn7(self.conv7(x)))
    x = F.relu(self.bn8(self.conv8(x)))
    x = self.pool(x)
    x = x.view(-1, 512)
    x = self.fc1(x)
    out = F.relu(x)
    return out
