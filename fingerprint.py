import glob
import cv2
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

class Fingerprints(Dataset):
    def __init__(self, finger, focus, img_dim=(416, 416)):
        self.img_dim = img_dim
        self.images = []
        self.labels = []
        finger = finger.split(' ')
        for img_path in glob.glob(DATASET_URL + f'/*{finger[0].capitalize()}_{"_".join([e for e in finger[1:]])}.BMP'):
            img_id = int((img_path.split('/')[-1]).split('_')[0])
            self.images.append(img_path)
            self.labels.append(1 if img_id == focus else 0)
        focus = self.images[int(np.where(self.labels)[0])]
        for o in range(len(self.labels) // 2):
            self.images.append(focus)
            self.labels.append(1)

    def __len__(self): 
        return len(self.labels)

    def __getitem__(self, idx):
        img_path, isfocus = self.images[idx], self.labels[idx]
        img = torch.from_numpy(cv2.resize(cv2.imread(img_path), self.img_dim))
        return img.permute(2, 0, 1).float(), torch.tensor([isfocus])

class ExampleImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 101 * 101, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.s = nn.Sigmoid()
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.s(x)
        return x

def train(dataloader, network, loss_fn, optimiser):
    for epoch in range(1):
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            X, y = batch
            optimiser.zero_grad()
            pred = network(X)
            loss = loss_fn(pred, y.float())
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            print(f'{i + 1} loss: {running_loss}')
            running_loss = 0.0

def test(testset, model, loss_fn):
    correct = 0
    with torch.no_grad():
        for data in testset:
            X, y = data
            pred = model(X)
            correct += (pred == y).sum().item()
    print(f'accuracy: {100 * correct // len(testset)}%')

DATASET_URL = 'data/socofing/real'
GENERIC_MODEL_URL = 'models/generic.pth'

batch_size = 30 

net = ExampleImageNet()
loss_fn = nn.MSELoss() 
optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

trainset = DataLoader(
        Fingerprints('left index finger', random.randint(1, 600)), 
        batch_size=batch_size, shuffle=True)
# net.load_state_dict(torch.load(GENERIC_MODEL_URL))
train(trainset, net, loss_fn, optimiser)
# torch.save(net.state_dict(), GENERIC_MODEL_URL)

# testset = DataLoader(
#         Fingerprints('left index finger', 100),
#         shuffle=True)
# net.load_state_dict(torch.load(GENERIC_MODEL_URL))
# test(testset, net, loss_fn)
