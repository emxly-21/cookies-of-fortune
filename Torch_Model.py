import torch
from torch import tensor
import torch.nn as nn
relu = nn.functional.relu


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        self.dense2 = nn.Linear(500, 50)
        self.dense3 = nn.Linear(50, 29)

        for layer in (self.dense2, self.dense3, self.conv1, self.conv2):
            nn.init.xavier_normal_(layer.weight, np.sqrt(2))
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        pool1 = nn.MaxPool2d(3, stride = 2))
        step1 = pool1(relu(self.conv1(x)))
        pool2 = nn.MaxUnpool2d(3, stride = 2)
        step2 = pool2(self.conv2(step1))
        print(step2.shape)
        flatten = step2.reshape(-1, 500)
        dense_layers = self.dense3(relu(self.dense2(flatten)))

        return dense_layers


