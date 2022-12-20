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

class Fingerprints(Dataset):
    def __init__(self, finger, focus, img_dim=(416, 416)):
        self.img_dim = img_dim
        self.images, self.labels = [], []
        finger = finger.split(' ')
        for img_path in glob.glob(DATASET_PATH + f'/*{finger[0].capitalize()}_{"_".join([e for e in finger[1:]])}.BMP'):
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
    def create_embedding(img, feature_extractor, embedding_dim):
        embedding = torch.randn(embedding_dim)
        with torch.no_grad():
            enc_out = feature_extractor(img)
            embedding = torch.cat((embedding, enc_out), 0)
        return embedding

    # @staticmethod
    # def feature_match(img, all_features):
    #     features = ImageProcess.extract_features(img)
    #     img_distances = scipy.spatial.distance.cdist(all_features, features.reshape(1, -1), 'cosine').reshape(-1) # cos_cdist
    #     topid = np.argsort(img_distances)[:1].tolist()
    #     return topid

    # @staticmethod
    # def compute_img_match(feature_extractor, img, embedding):
    #     with torch.no_grad():
    #         img_embedding = feature_extractor(img).detach().numpy()
    #     img_embedding = img_embedding.reshape((img_embedding.shape[0], -1))
    #     knn = NearestNeighbors(n_neighbours=1, metric='cosine')
    #     knn.fit(embedding)
    #     _, idx = knn.neighbors(img_embedding)
    #     idx = idx.tolist()
    #     return idx

def imshow(img):
    imgplot = plt.imshow(img.astype('uint8'), cmap='gray')
    plt.show()

class FingerprintFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)

    def forward(self, x):
        x = [ImageProcess.morph_op(img) for img in x]
        # x = [ImageProcess.g_filter(img) for img in x]

        x = torch.from_numpy(np.array(x)).float()
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        return x

fd = nn.Sequential(
        nn.ConvTranspose2d(256, 128, 2, stride=2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(128, 64, 2, stride=2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 32, 2, stride=2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(32, 16, 2, stride=2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(16, 1, 2, stride=2),
        nn.ReLU(inplace=True)
        )

class FingerprintModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.s = nn.Sigmoid()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.s(x)
        return x

def train(epochs):
    def train_feature_extractor(epochs, tmp_trainset):
        feature_extractor = FingerprintFeatureExtractor()
        loss_fn = nn.MSELoss()
        autoencoder_params = list(feature_extractor.parameters()) + list(fd.parameters())
        optimiser = optim.Adam(autoencoder_params, lr=1e-3)
        for e in range(epochs):
            for i, (img, _) in enumerate(tmp_trainset):
                optimiser.zero_grad()
                enc_out = feature_extractor(img)
                dec_out = fd(enc_out)
                loss = loss_fn(dec_out, img.float())
                loss.backward()
                optimiser.step()
                print(f'{i + 1} loss: {loss.item()}')
        return feature_extractor
    net = FingerprintModel()
    loss_fn = nn.MSELoss()
    optimiser = optim.Adam(net.parameters(), lr=1e-3)
    batch_size = 30 
    trainset = DataLoader(Fingerprints('left index finger', random.randint(1, 600)), batch_size=batch_size, shuffle=True)
    feature_extractor = train_feature_extractor(epochs, trainset)
    for e in range(epochs):
        for i, batch in enumerate(trainset):
            X, y = batch
            optimiser.zero_grad()
            X = [ImageProcess.create_embedding(img, feature_extractor, EMBEDDING_SHAPE[1:]).detach().numpy() for img in X]
            X = torch.from_numpy(np.array(X))
            pred = net(X)
            loss = loss_fn(pred, y.float())
            loss.backward()
            optimiser.step()
            print(f'{i + 1} loss: {loss.item()}')
    return feature_extractor, net

DATASET_PATH = 'data/socofing/real'
MODEL_PATH = 'models/'
EMBEDDING_SHAPE = (1, 256, 13, 13)


def test(fe, net):
    with torch.no_grad():
        testset = DataLoader(Fingerprints('left index finger', 100), batch_size=1, shuffle=True)
        for i, batch in enumerate(testset):
            X, y = batch
            print(X.shape)
            X = [ImageProcess.create_embedding(img, feature_extractor, EMBEDDING_SHAPE[1:]).detach().numpy() for img in X]
            X = torch.from_numpy(np.array(X))
            pred = net(X)
            print(f'{i + 1} pred: {pred} actual: {y}')
feature_extractor, net = train(1)
test(feature_extractor, net)
