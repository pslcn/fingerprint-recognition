import glob
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

class TestFingerprints(Dataset):
    def __init__(self, finger, focus, img_dim=(416, 416)):
        self.img_dim = img_dim
        self.images, self.labels = [], []
        finger = finger.split(' ')
        for img_path in glob.glob(DATASET_URL + f'/*{finger[0].capitalize()}_{"_".join([e for e in finger[1:]])}.BMP'):
            img_id = int((img_path.split('/')[-1]).split('_')[0])
            self.images.append(cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), self.img_dim)[np.newaxis, :, :])
            self.labels.append(1 if img_id == focus else 0)
        focus = self.images[int(np.where(self.labels)[0])]
        for o in range(len(self.labels) // 2):
            self.images.append(focus)
            self.labels.append(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor([self.labels[idx]])

class ImageProcess:
    @staticmethod
    def morph_op(img):
        cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype('uint8')
        ret = cv2.erode(img.numpy(), cross, iterations=1)
        return cv2.dilate(ret, cross, iterations=2)

    @staticmethod
    def g_filter(img):
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
        for kern in filters: np.maximum(newimage, cv2.filter2D(img, depth, kern), newimage)
        return newimage

    @staticmethod
    def extract_features(img, vector_size=32):
        try:
            alg = cv2.KAZE_create()
            kps = alg.detect(img)
            kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
            kps, dsc = alg.compute(img, kps)
            dsc = dsc.flatten()
            needed_size = (vector_size * 64)
            if dsc.size < needed_size:
                dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        except cv2.error as e:
            return None
        return dsc

    @staticmethod
    def feature_match(img, all_features):
        features = ImageProcess.extract_features(img)
        img_distances = scipy.spatial.distance.cdist(all_features, features.reshape(1, -1), 'cosine').reshape(-1) # cos_cdist
        topid = np.argsort(img_distances)[:1].tolist()
        return topid

def show_transform(img1, img2):
    imgs = np.hstack((img1, img2, np.abs(np.subtract(img1, img2))))
    plt.imshow(imgs.astype('uint8'), cmap='nipy_spectral')
    plt.show()

class FingerprintModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 101 * 101, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.s = nn.Sigmoid()

    def forward(self, x):
        # imgplot = plt.imshow(x[0][0].numpy().astype('uint8'), cmap='gray')
        # plt.show()

        # Morphological operations
        # x = np.array([ImageProcess.morph_op(img) for img in x])
        x = [ImageProcess.morph_op(img) for img in x]
        # imgplot = plt.imshow(x[0][0].astype('uint8'), cmap='gray')
        # plt.show()

        # Gabor filtering for image enhancement
        # x = np.array([ImageProcess.g_filter(img) for img in x])
        x = [ImageProcess.g_filter(img) for img in x]
        # imgplot = plt.imshow(x[0][0].astype('uint8'), cmap='gray')
        # plt.show()

        # Feature extraction
        # features = np.array([ImageProcess.extract_features(img[0]) for img in x])

        # Matching image
        # focus_features = ImageProcess.extract_features(focus_img)
        # imgplot = plt.imshow(torch.squeeze(torch.tensor(x[ImageProcess.feature_match(focus_features, features)])).numpy().astype('uint8'), cmap='gray')
        # plt.show()

        x = np.array(x)
        x = torch.from_numpy(x).float()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.s(x)
        return x

DATASET_URL = 'data/socofing/real'
GENERIC_MODEL_URL = 'models/generic.pth'

net = FingerprintModel()
loss_fn = nn.MSELoss() 
optimiser = optim.SGD(net.parameters(), lr=0.01)

batch_size = 30 
trainset = DataLoader(TestFingerprints('left index finger', random.randint(1, 600)), batch_size=batch_size, shuffle=True)

def train(epochs, dataloader, network, loss_fn, optimiser):
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            X, y = batch
            optimiser.zero_grad()
            pred = network(X)
            loss = loss_fn(pred, y.float())
            loss.backward()
            optimiser.step()
            print(f'{i + 1} loss: {loss.item()}')

# net.load_state_dict(torch.load(GENERIC_MODEL_URL))
train(1, trainset, net, loss_fn, optimiser)
# torch.save(net.state_dict(), GENERIC_MODEL_URL)
