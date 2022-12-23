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

def imshow(img):
    imgplot = plt.imshow(img.astype('uint8'), cmap='gray')
    plt.show()

def train(epochs, trainset):
    for e in range(epochs):
        for i, batch in enumerate(trainset):
            X, y = batch
            X = torch.from_numpy(np.array(X)).float()

            # imshow(X[0][0].detach().numpy())
            # X = [ImageProcess.morph_op(img) for img in X]
            # X = [ImageProcess.gabor_filter(img) for img in X]

            features = extract_features(X)

            # optimiser.zero_grad()
            # pred = net(X)
            # loss = loss_fn(pred, y)
            # loss.backward()
            # optimiser.step()
            # print(f'epoch: {e + 1} batch: {i + 1} loss: {loss.item()}')

def extract_features(imgs):
    with torch.no_grad():
        features = [net(img).detach().numpy().reshape(-1) for img in imgs]
    return np.array(features)

DATASET_PATH = 'data/socofing/real'
MODEL_PATH = 'models/'

model = models.vgg16(weights='DEFAULT')
net = FeatureExtractor(model)

# loss_fn = nn.MSELoss()
# optimiser = optim.Adam(net.parameters(), lr=0.01)

batch_size = 30

train_fingerprints = Fingerprints('left index finger', random.randint(1, 600))
train_focus = train_fingerprints.get_dataset_focus()

train(1, DataLoader(train_fingerprints, batch_size=batch_size, shuffle=True))
