import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class MultipleRegression(nn.Module):
    def __init__(self, num_features):
        super(MultipleRegression, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.fc2 = nn.Linear(64, 128)
#         self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(128, 1)

        self.act = nn.Sigmoid()

    def forward(self, inputs):
        x = self.act(self.fc1(inputs))
        x = self.act(self.fc2(x))
#         x = self.act(self.fc3(x))
        x = self.output(x)

        return x

    def predict(self, test_inputs):
        x = self.act(self.fc1(test_inputs))
        x = self.act(self.fc2(x))
#         x = self.act(self.fc3(x))
        x = self.output(x)

        return x
