import torch
from torch import nn

class SimpleNet(nn.Module):
    def __init__(self, focus):
        super().__init__()
        self.focus = focus
        self.fe = nn.Sequential(
            nn.Conv2d(1, 3, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(1),
            nn.Linear(16 * 52 * 52, 4096, bias=True)
        )
        self.conclude = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fe(x)
        focus = self.fe(self.focus)
        x = torch.sub(x, focus)
        x = self.conclude(x)
        return x
