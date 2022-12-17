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
    def __init__(self, finger, img_dim=(416, 416)):
        self.img_dim = img_dim
        self.data = []
        finger = finger.split(' ')
        for img_path in glob.glob(DATASET_URL + f'/*{finger[0].capitalize()}_{"_".join([e for e in finger[1:]])}.BMP'):
            img_id = int((img_path.split('/')[-1]).split('_')[0])
            self.data.append([img_path, img_id])

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        img_path, img_id = self.data[idx]
        img = torch.from_numpy(cv2.resize(cv2.imread(img_path), self.img_dim))
        return img.permute(2, 0, 1).float(), img_id

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
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.s(x)
        return x

DATASET_URL = 'data/socofing/real'

fingerprints = Fingerprints('left index finger')
batch_size = 100 
trainset = DataLoader(fingerprints, batch_size=batch_size, shuffle=True)

def train(dataloader, network, loss_fn, optimiser):
    for epoch in range(1):
        running_loss = 0.0
        for i, (X, y) in enumerate(dataloader):
            X, y = X, y 
            fid = random.choice(y) 
            y = torch.tensor([[1 if l == fid else 0] for l in y])
            optimiser.zero_grad()
            pred = network(X)
            loss = loss_fn(pred, y.float())
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            print(f'[{i + 1} loss: {running_loss}]')
            running_loss = 0.0

def test(testset, model, loss_fn):
    size = len(testset.dataset)
    batches = len(testset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for fid in range(1):
            for X, y in testset:
                y = torch.tensor([[1 if l == torch.tensor([fid]) else 0] for l in y]) 
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= batches
            # correct /= size
            # print(f'accuracy: {(100 * correct):>0.1f}%, avg loss: {test_loss:>8f}')

net = ExampleImageNet()
loss_fn = nn.MSELoss() 
optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

MODEL_PATH = 'models/net.pth'

train(trainset, net, loss_fn, optimiser)
# torch.save(net.state_dict(), MODEL_PATH)

# testset = trainset
# net.load_state_dict(torch.load(MODEL_PATH))
# test(testset, net, loss_fn)
