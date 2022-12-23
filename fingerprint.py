import glob
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

class Fingerprints(Dataset):
    def __init__(self, finger, focus, img_dim=(416, 416)):
        self.images, self.labels = [], []
        finger = finger.split(' ')
        for img_path in glob.glob(DATASET_PATH + f'/*{finger[0].capitalize()}_{"_".join([e for e in finger[1:]])}.BMP'):
            img_id = int((img_path.split('/')[-1]).split('_')[0])
            self.images.append(cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), img_dim)[np.newaxis, :, :])
            self.labels.append(1 if img_id == focus else 0)
        self.focus = self.images[int(np.where(self.labels)[0])]
        for o in range(len(self.labels) // 2):
            self.images.append(self.focus)
            self.labels.append(1)

    def get_dataset_focus(self):
        return self.focus

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor([self.labels[idx]]).float()

class ImageProcess:
    @staticmethod 
    def morph_op(img):
        cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype('uint8')
        ret = cv2.erode(img.numpy(), cross, iterations=1)
        return cv2.dilate(ret, cross, iterations=1)

    @staticmethod
    def gabor_filter(img):
        filters = []
        num_filters = 16
        ksize = 35
        sigma = 3.0
        lambd = 10.0
        gamma = 0.5
        psi = 0
        for theta in np.arange(0, np.pi, np.pi / num_filters):
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
            kern /= 1.0 * kern.sum()
            filters.append(kern)
        newimage = np.zeros_like(img)
        depth = -1
        for kern in filters:
            np.maximum(newimage, cv2.filter2D(img, depth, kern), newimage) 
        return newimage

class FingerprintModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 102 * 102, 120)
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

def imshow(img):
    imgplot = plt.imshow(img.astype('uint8'), cmap='gray')
    plt.show()

def train(epochs, trainset):
    for e in range(epochs):
        for i, batch in enumerate(trainset):
            X, y = batch

            # imshow(X[0][0].detach().numpy())
            # X = [ImageProcess.morph_op(img) for img in X]
            # X = [ImageProcess.gabor_filter(img) for img in X]

            X = torch.from_numpy(np.array(X)).float()

            optimiser.zero_grad()
            pred = net(X.float())
            loss = loss_fn(pred, y)
            loss.backward()
            optimiser.step()
            print(f'epoch: {e + 1} batch: {i + 1} loss: {loss.item()}')

DATASET_PATH = 'data/socofing/real'
MODEL_PATH = 'models/'

net = FingerprintModel()
loss_fn = nn.MSELoss()
optimiser = optim.Adam(net.parameters(), lr=0.01)

batch_size = 30

train_fingerprints = Fingerprints('left index finger', random.randint(1, 600))
train_focus = train_fingerprints.get_dataset_focus()

train(4, DataLoader(train_fingerprints, batch_size=batch_size, shuffle=True))
