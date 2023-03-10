import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import imgs
from imgs import Fingerprints
from model import SimpleCNN

MODEL_PATH = 'models/'
SAVE_PATH = MODEL_PATH + 'net.pth'

# """ Training and Testing:
DATASET_PATH = 'data/socofing/'
TRAINSET_PATH = DATASET_PATH + 'real'
TESTSET_PATH = lambda difficulty : DATASET_PATH + 'altered/' + difficulty

def imshow(img):
    imgplot = plt.imshow(img.astype('uint8'), cmap='gray')
    plt.show()

def train(epochs, trainset_path, finger, focus_id, focus, save_path, batch_size=30):
    fingerprints = Fingerprints(trainset_path, finger, focus_id) 
    fingerprints.pad_with_focus(focus)
    net = SimpleCNN()
    loss_fn = nn.MSELoss()
    optimiser = optim.Adam(net.parameters(), lr=0.01)
    for e in range(epochs):
        for i, (X, y) in enumerate(DataLoader(fingerprints, batch_size=batch_size, shuffle=True)):
            focus = torch.from_numpy(np.array(focus[random.randint(0, len(focus) - 1)])).float()[None]
            # Preprocessing is done on X as np arrays
            X = torch.from_numpy(np.array(X)).float() # Model uses tensors
            optimiser.zero_grad()
            pred = net(X, focus)
            loss = loss_fn(pred, y)
            loss.backward()
            optimiser.step()
            print(f'epoch: {e + 1} batch: {i + 1} loss: {loss.item()}')
    torch.save(net.state_dict(), save_path)

def test(testset_path, finger, focus_id, focus, load_path):
    fingerprints = Fingerprints(testset_path, finger, focus_id)
    fingerprints.pad_with_focus(focus)
    net = SimpleCNN()
    net.load_state_dict(torch.load(load_path))
    batch_size = 30
    net.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(DataLoader(fingerprints, batch_size=batch_size, shuffle=True)):
            focus = torch.from_numpy(np.array(focus[random.randint(0, len(focus) - 1)])).float()[None]
            X = torch.from_numpy(np.array(X)).float()
            pred = net(X, focus)
            num_correct = sum([1 if(abs(a - b) <= 0.1) else 0 for a, b in zip(pred.detach().numpy(), y.detach().numpy())])
            print(f'batch: {i + 1} accuracy: {(num_correct / batch_size) * 100}%')

FINGER = 'left index finger'.split(' ')
FOCUS_ID = random.randint(1, 600)
FOCUS = imgs.get_focus_fingerprints(TESTSET_PATH('hard'), FINGER, FOCUS_ID)

# train(2, TRAINSET_PATH, FINGER, FOCUS_ID, FOCUS, SAVE_PATH)
test(TESTSET_PATH('hard'), FINGER, FOCUS_ID, FOCUS, SAVE_PATH)
# """

"""
import sys
import cv2

saved = cv2.resize(cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE), (416, 416))[np.newaxis, :, :]
saved = torch.from_numpy(np.array(saved)).float()[None]
query = cv2.resize(cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE), (416, 416))[np.newaxis, :, :]
net = SimpleCNN()
net.load_state_dict(torch.load(SAVE_PATH))
net.eval()
with torch.no_grad():
    X = torch.from_numpy(np.array(query)).float()[None]
    pred = net(X, saved)
    print(f'pred: {pred.item()}')
"""
