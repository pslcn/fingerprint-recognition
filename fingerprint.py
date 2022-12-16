import glob
import cv2
import os
import imageio.v2 as imageio
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torch.optim as optim
import torch.nn.functional as F

DATASET_URL = 'data/socofing/real'

# def get_img(name):
#     return np.asarray(imageio.imread(os.path.join(DATASET_URL, name)))
# fp = get_img('100__M_Left_index_finger.BMP')

class Fingerprints(Dataset):
    def __init__(self):
        self.data = []
        for img_path in glob.glob(DATASET_URL + '/*.BMP'):
            img_id = int((img_path.split('_')[0]).split('/')[-1])
            self.data.append([img_path, img_id])
        self.img_dim = (416, 416)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, img_id = self.data[idx]
        img = torch.from_numpy(cv2.resize(cv2.imread(img_path), self.img_dim))
        img = img.permute(2, 0, 1)
        # img_id = torch.tensor([img_id])
        return img.float(), img_id

fingerprints = Fingerprints()
print('The shape of tensor for 50th image in train dataset: ',fingerprints[49][0].shape)
print('The label for 50th image in train dataset: ',fingerprints[49][1])

batch_size = 4
trainset = DataLoader(fingerprints, batch_size=batch_size, shuffle=True)

class ExampleImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 101 * 101, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.reshape(x, (-1,)) # Reshape [4, 1] into [4]
        return x

net = ExampleImageNet()

# loss_fn = nn.CrossEntropyLoss() 
loss_fn = nn.L1Loss()
optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def train(dataloader, model, loss_fn, optimiser):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X, y
        pred = model(X)
        loss = loss_fn(pred, y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

for t in range(1): train(trainset, net, loss_fn, optimiser)
print('Done!')
