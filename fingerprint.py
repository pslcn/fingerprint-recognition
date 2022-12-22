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
        self.focus = self.images[int(np.where(self.labels)[0])]
        for o in range(len(self.labels) // 2):
            self.images.append(self.focus)
            self.labels.append(1)

    def get_dataset_focus(self):
        return self.focus 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor([self.labels[idx]])

# class ImageProcess:
#     @staticmethod
#     def apply_morphological_op(img):
#         cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype('uint8')
#         ret = cv2.erode(img.numpy(), cross, iterations=1)
#         return cv2.dilate(ret, cross, iterations=1)
# 
#     @staticmethod
#     def apply_gabor_filters(img):
#         filters = []
#         num_filters = 16
#         ksize = 35
#         sigma = 3.0
#         lambd = 10.0
#         gamma = 0.5
#         psi = 0
#         for theta in np.arange(0, np.pi, np.pi / num_filters): 
#             kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F) 
#             kern /= 1.0 * kern.sum()
#             filters.append(kern)
#         newimage = np.zeros_like(img)
#         depth = -1
#         for kern in filters: np.maximum(newimage, cv2.filter2D(img, depth, kern), newimage)
#         return newimage

    # @staticmethod
    # def extract_visual_descriptors(img, vector_size=32):
    #     try:
    #         alg = cv2.KAZE_create()
    #         kps = alg.detect(img)
    #         kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
    #         kps, dsc = alg.compute(img, kps)
    #         dsc = dsc.flatten()
    #         needed_size = (vector_size * 64)
    #         if dsc.size < needed_size:
    #             dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    #     except cv2.error as e:
    #         return None
    #     return dsc
    #
    # @staticmethod
    # def extract_fast_descriptors(img):
    #     fast = cv2.FastFeatureDetector_create()
    #     keypoints_with_nonmax = fast.detect(img, None)
    #     img_with_nonmax = np.copy(img)
    #     return cv2.drawKeypoints(img, keypoints_with_nonmax, img_with_nonmax, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #
    # @staticmethod
    # def create_embedding(img, feature_extractor, embedding_dim):
    #     embedding = torch.randn(embedding_dim)
    #     with torch.no_grad():
    #         enc_out = feature_extractor(img)
    #         embedding = torch.cat((embedding, enc_out), 0)
    #     return embedding

    # @staticmethod
    # def feature_vectors_cos_dist(img, feature_vectors):
    #     features = ImageProcess.extract_features(img)
    #     img_distances = scipy.spatial.distance.cdist(feature_vectors, features.reshape(1, -1), 'cosine').reshape(-1) # cos_cdist
    #     return np.argsort(img_distances)[:1].tolist()
    #
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

# class FingerprintFeatureExtractor(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.pool = nn.MaxPool2d(2)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
#         self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
#         self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
# 
#     def forward(self, x):
#         x = [ImageProcess.apply_morphological_op(img) for img in x]
#         # x = [ImageProcess.g_filter(img) for img in x]
# 
#         x = torch.from_numpy(np.array(x)).float()
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = self.pool(self.relu(self.conv3(x)))
#         x = self.pool(self.relu(self.conv4(x)))
#         x = self.pool(self.relu(self.conv5(x)))
#         return x
# 
# fd = nn.Sequential(
#         nn.ConvTranspose2d(256, 128, 2, stride=2),
#         nn.ReLU(inplace=True),
#         nn.ConvTranspose2d(128, 64, 2, stride=2),
#         nn.ReLU(inplace=True),
#         nn.ConvTranspose2d(64, 32, 2, stride=2),
#         nn.ReLU(inplace=True),
#         nn.ConvTranspose2d(32, 16, 2, stride=2),
#         nn.ReLU(inplace=True),
#         nn.ConvTranspose2d(16, 1, 2, stride=2),
#         nn.ReLU(inplace=True)
#         )
# 
# def train_feature_extractor(epochs, tmp_trainset):
#     feature_extractor = FingerprintFeatureExtractor()
#     loss_fn = nn.MSELoss()
#     autoencoder_params = list(feature_extractor.parameters()) + list(fd.parameters())
#     optimiser = optim.Adam(autoencoder_params, lr=1e-3) 
#     for e in range(epochs):
#         for i, (img, _) in enumerate(tmp_trainset):
#             optimiser.zero_grad()
#             enc_out = feature_extractor(img)
#             dec_out = fd(enc_out)
#             loss = loss_fn(dec_out, img.float())
#             loss.backward()
#             optimiser.step()
#             print(f'epoch: {e + 1} {i + 1} loss: {loss.item()}')
#     return feature_extractor
# 
# def train(epochs):
#     net = FingerprintModel()
#     loss_fn = nn.MSELoss()
#     optimiser = optim.Adam(net.parameters(), lr=1e-3)
#     batch_size = 30 
#     trainset = DataLoader(Fingerprints('left index finger', random.randint(1, 600)), batch_size=batch_size, shuffle=True)
#     feature_extractor = train_feature_extractor(epochs, trainset)
#     for e in range(epochs):
#         for i, batch in enumerate(trainset):
#             X, y = batch
#             optimiser.zero_grad()
#             X = [ImageProcess.create_embedding(img, feature_extractor, EMBEDDING_SHAPE[1:]).detach().numpy() for img in X]
#             X = torch.from_numpy(np.array(X))
#             pred = net(X)
#             loss = loss_fn(pred, y.float())
#             loss.backward()
#             optimiser.step()
#             print(f'epoch: {e + 1} {i + 1} loss: {loss.item()}')
#     return feature_extractor, net

DATASET_PATH = 'data/socofing/real'
MODEL_PATH = 'models/'

# EMBEDDING_SHAPE = (1, 256, 13, 13)

def imshow(img):
    imgplot = plt.imshow(img.astype('uint8'))
    plt.show()

def cv2_brute_force_match(saved_img, saved_keypoints, saved_descriptor, query_img, query_keypoints, query_descriptor):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(saved_descriptor, query_descriptor)
    matches = sorted(matches, key=lambda x: x.distance)
    img_showing_matches = cv2.drawMatches(query_img, query_keypoints, saved_img, saved_keypoints, matches, saved_img, flags=2)
    num_matches = np.array([len(matches)])
    return num_matches

def orb_compare(saved, query):
    orb = cv2.ORB_create()
    saved_keypoints, saved_descriptor = orb.detectAndCompute(saved, None)
    query_keypoints, query_descriptor = orb.detectAndCompute(query, None)
    keypoints_without_size = np.copy(saved)
    cv2.drawKeypoints(saved, saved_keypoints, keypoints_without_size, color=(0, 0, 255))
    return saved_keypoints, saved_descriptor, query_keypoints, query_descriptor

train_fingerprints = Fingerprints('left index finger', random.randint(1, 600))
train_focus = train_fingerprints.get_dataset_focus()[0]

def train(epochs, trainset):
    for e in range(epochs):
        for i, batch in enumerate(trainset):
            X, y = batch

            img = X[0][0].detach().numpy()
            saved_keypoints, saved_descriptor, query_keypoints, query_descriptor = orb_compare(train_focus, img)
            m = cv2_brute_force_match(train_focus, saved_keypoints, saved_descriptor, img, query_keypoints, query_descriptor)
            # corrcoef = np.corrcoef(m, y[0].detach().numpy())

train(1, DataLoader(train_fingerprints, batch_size=30, shuffle=True))
