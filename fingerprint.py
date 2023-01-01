import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import imgs
from imgs import Fingerprints

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

def imshow(img):
    imgplot = plt.imshow(img.astype('uint8'), cmap='gray')
    plt.show()

def train(epochs, save_path):
    fingerprints = Fingerprints(TRAINSET_PATH, FINGER, FOCUS_ID) 
    net = SimpleNet(torch.from_numpy(np.array(FOCUS)).float()[None])
    batch_size = 30
    net.train()
    fingerprints.pad_with_focus(FOCUS)
    loss_fn = nn.MSELoss()
    optimiser = optim.Adam(net.parameters(), lr=0.01)
    for e in range(epochs):
        for i, (X, y) in enumerate(DataLoader(fingerprints, batch_size=batch_size, shuffle=True)):
            X = torch.from_numpy(np.array(X)).float()
            optimiser.zero_grad()
            pred = net(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimiser.step()
            print(f'epoch: {e + 1} batch: {i + 1} loss: {loss.item()}')
    torch.save(net.state_dict(), save_path)

def test(load_path):
    fingerprints = Fingerprints(TESTSET_PATH('hard'), FINGER, FOCUS_ID)
    net = SimpleNet(torch.from_numpy(np.array(FOCUS)).float()[None])
    net.load_state_dict(torch.load(load_path))
    batch_size = 30
    net.eval()
    fingerprints.pad_with_focus(FOCUS)
    with torch.no_grad():
        for i, (X, y) in enumerate(DataLoader(fingerprints, batch_size=batch_size, shuffle=True)):
            X = torch.from_numpy(np.array(X)).float()
            pred = net(X)
            num_correct = sum([1 if(abs(a - b) <= 0.1) else 0 for a, b in zip(pred.detach().numpy(), y.detach().numpy())])
            print(f'batch: {i + 1} accuracy: {(num_correct / batch_size) * 100}%')

# """ For training and testing:
DATASET_PATH = 'data/socofing/'
TRAINSET_PATH = DATASET_PATH + 'real'
TESTSET_PATH = lambda difficulty : DATASET_PATH + 'altered/' + difficulty

FINGER = 'left index finger'.split(' ')
FOCUS_ID = random.randint(1, 600)
FOCUS = imgs.get_focus_fingerprint(TRAINSET_PATH, FINGER, FOCUS_ID)
# """

MODEL_PATH = 'models/'
SAVE_PATH = MODEL_PATH + 'net.pth'

test(SAVE_PATH)

"""
import sys

saved = cv2.resize(cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE), (416, 416))[np.newaxis, :, :]
query = cv2.resize(cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE), (416, 416))[np.newaxis, :, :]
net = SimpleNet(torch.from_numpy(np.array(saved)).float()[None])
net.load_state_dict(torch.load(SAVE_PATH))
net.eval()
with torch.no_grad():
    X = torch.from_numpy(np.array(query)).float()[None]
    pred = net(X)
    print(f'pred: {pred.item()}')
"""
