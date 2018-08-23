import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(BasicQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class VisualQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(VisualQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        if len(state_size)==5:
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 3, 3), stride=(1,3,3))
            self.bn1 = nn.BatchNorm3d(64)
            self.conv2 = nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1,3,3))
            self.bn2 = nn.BatchNorm3d(128)
            self.conv3 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1,3,3))
            self.bn3 = nn.BatchNorm3d(128)
        else:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=3)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=3)
            self.bn3 = nn.BatchNorm2d(64)
        conv_out_size = self._get_conv_out_size(state_size)
        fc = [conv_out_size, 1024]
        self.fc1 = nn.Linear(fc[0], fc[1])
        self.fc2 = nn.Linear(fc[1], action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self._cnn(state)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # generate input sample and forward to get shape
    def _get_conv_out_size(self, shape):
        x = torch.rand(shape)
        x = self._cnn(x)
        n_size = x.data.view(1, -1).size(1)
        print('Convolution output size:', n_size)
        return n_size

    def _cnn(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x
