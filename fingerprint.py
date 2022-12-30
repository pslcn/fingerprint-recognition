import glob
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models

class Fingerprints(Dataset):
    def __init__(self, path, finger, focus, img_dim=(416, 416)):
        self.images, self.labels = [], []
        for img_path in glob.glob(path + f'/*{finger[0].capitalize()}_{"_".join([e for e in finger[1:]])}*.BMP'):
            img_id = int((img_path.split('/')[-1]).split('_')[0])
            self.images.append(cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), img_dim)[np.newaxis, :, :])
            self.labels.append(1 if img_id == focus else 0)

    def pad_with_focus(self):
        for o in range(len(self.labels) // 4):
            self.images.append(FOCUS)
            self.labels.append(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor([self.labels[idx]]).float()

class ImageProcess:
    @staticmethod 
    def morph_op(img):
        cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype('uint8')
        return cv2.dilate(cv2.erode(img.numpy(), cross, iterations=1), cross, iterations=1)

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

    @staticmethod
    def vector_features_cosdist(v1, v2):
        return scipy.spatial.distance.cdist(v1, v2, 'cosine')

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
    fingerprints.pad_with_focus()
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
    fingerprints.pad_with_focus()
    with torch.no_grad():
        for i, (X, y) in enumerate(DataLoader(fingerprints, batch_size=batch_size, shuffle=True)):
            X = torch.from_numpy(np.array(X)).float()
            pred = net(X)
            num_correct = sum([1 if(abs(a - b) <= 0.1) else 0 for a, b in zip(pred.detach().numpy(), y.detach().numpy())])
            print(f'batch: {i + 1} accuracy: {(num_correct / batch_size) * 100}%')

def get_focus_fingerprint(path, img_dim=(416, 416)):
    for img_path in glob.glob(path + f'/*{FINGER[0].capitalize()}_{"_".join([e for e in FINGER[1:]])}*.BMP'):
        img_id = int((img_path.split('/')[-1]).split('_')[0])
        if img_id == FOCUS_ID:
            return cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), img_dim)[np.newaxis, :, :]
    return None

DATASET_PATH = 'data/socofing/'
TRAINSET_PATH = DATASET_PATH + 'real'
TESTSET_PATH = lambda difficulty : DATASET_PATH + 'altered/' + difficulty

MODEL_PATH = 'models/'
SAVE_PATH = MODEL_PATH + 'net.pth'

FINGER = 'left index finger'.split(' ')
FOCUS_ID = random.randint(1, 600)
FOCUS = get_focus_fingerprint(TRAINSET_PATH)

# train(2, MODEL_PATH + 'test.pth')
test(SAVE_PATH)
