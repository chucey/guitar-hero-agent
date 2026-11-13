'''
Define the vision critic model, which takes in a sequence of frames and outputs a value estimate for the current state.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionCritic(nn.Module):
    def __init__(self, input_shape):
        super(VisionCritic, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self._get_conv_output(input_shape), 512)
        self.fc2 = nn.Linear(512, 1)

    def _get_conv_output(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(torch.prod(torch.tensor(o.size())))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x