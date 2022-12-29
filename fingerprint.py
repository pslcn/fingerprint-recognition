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
    def __init__(self, finger, focus, img_dim=(416, 416)):
        self.images, self.labels = [], []
        finger = finger.split(' ')
        for img_path in glob.glob(DATASET_PATH + f'/*{finger[0].capitalize()}_{"_".join([e for e in finger[1:]])}.BMP'):
            img_id = int((img_path.split('/')[-1]).split('_')[0])
            self.images.append(cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), img_dim)[np.newaxis, :, :])
            self.labels.append(1 if img_id == focus else 0)
        self.focus = self.images[int(np.where(self.labels)[0])]

    def pad_with_focus(self):
        for o in range(len(self.labels) // 2):
            self.images.append(self.focus)
            self.labels.append(1)

    def get_dataset_focus(self):
        return self.focus

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor([self.labels[idx]]).float()

class FingerprintFeatures(Dataset):
    def __init__(self, features_path):
        self.features, self.labels = np.load(features_path)

    def pad_with_focus(self):
        self.focus = self.features[int(np.where(self.labels)[0])]
        for o in range(len(self.labels) // 2):
            self.features.append(self.focus)
            self.labels.append(1)

    def get_dataset_focus(self):
        return self.focus

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], torch.tensor([self.labels[idx]]).float()

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

    @staticmethod
    def vector_features_cosdist(v1, v2):
        distances = scipy.spatial.distance.cdist(v1, v2, 'cosine')
        return distances

    @staticmethod
    def simple_vector_dist(v1, v2):
        return np.array([np.subtract(a, b) for a, b in zip(v1, v2)])

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()

        self.features = list(model.features)

        self.features = nn.Sequential(
                nn.Conv2d(1, 3, 3),
                *self.features)
        self.pooling = model.avgpool
        self.fc = model.classifier[0]

    def forward(self, x):
        x = self.pooling(self.features(x))

        x = torch.flatten(x, 0)
        x = self.fc(x)

        return x

class VectorLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 84)
        self.fc4 = nn.Linear(84, 1)
        self.s = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.s(x)
        return x

def imshow(img):
    imgplot = plt.imshow(img.astype('uint8'), cmap='gray')
    plt.show()

def train(epochs, model, trainset):
    focus_features = np.array([np.squeeze(extract_features(focus), axis=0) for t in range(batch_size)])

    for e in range(epochs):
        for i, batch in enumerate(trainset):
            X, y = batch
            X = torch.from_numpy(np.array(X)).float()

            # imshow(X[0][0].detach().numpy())

            # X = [ImageProcess.morph_op(img) for img in X]
            # X = [ImageProcess.gabor_filter(img) for img in X]

            X = extract_features(X)

            # print(f'extract_features at 0: {X[0]}')

            X = ImageProcess.simple_vector_dist(focus_features, X)

            # print(f'simple_vector_dist at 0: {X[0]}')

            # dists = np.array([np.squeeze(ImageProcess.vector_features_cosdist(v1[np.newaxis], v2[np.newaxis]), axis=1) for v1, v2 in zip(features, focus_features)])
            # for i in range(batch_size): print(f'dists[{i}]: {dists[i]} label: {y[i]}')

            optimiser.zero_grad()
            pred = model(torch.from_numpy(X))
            loss = loss_fn(pred, y)
            loss.backward()
            optimiser.step()
            print(f'epoch: {e + 1} batch: {i + 1} loss: {loss.item()}')

def extract_features(imgs):
    with torch.no_grad():
        return np.array([fe(img).detach().numpy().reshape(-1) for img in imgs])

def test(model, testset):
    with torch.no_grad():
        for X, y in testset:
            pred = model(torch.from_numpy(X))
            print('pred: {pred} actual: {y}')

DATASET_PATH = 'data/socofing/real'
MODEL_PATH = 'models/'

SAVE_PATH = MODEL_PATH + 'net.pth'

# model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = models.vgg16(weights='DEFAULT')
model.eval()
for param in model.parameters():
    param.requires_grad = False

fe = FeatureExtractor(model)

net = VectorLinearNet()
# net.load_state_dict(torch.load(SAVE_PATH))

fingerprints = Fingerprints('left index finger', random.randint(1, 600))
focus = torch.from_numpy(np.array(fingerprints.get_dataset_focus())).float()[None]

# Training
batch_size = 30

net.train()
fingerprints.pad_with_focus()

loss_fn = nn.MSELoss()
optimiser = optim.Adam(net.parameters(), lr=0.01)

train(1, net, DataLoader(fingerprints, batch_size=batch_size, shuffle=True))
torch.save(net.state_dict(), SAVE_PATH)

# Testing
# net.load_state_dict(torch.load(SAVE_PATH))
# net.eval()
# 
# test(net, DataLoader(fingerprints, batch_size=1, shuffle=True))
