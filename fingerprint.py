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
from model import SimpleNet

MODEL_PATH = 'models/'
SAVE_PATH = MODEL_PATH + 'net.pth'

# """ Training and Testing:
def imshow(img):
    imgplot = plt.imshow(img.astype('uint8'), cmap='gray')
    plt.show()

def train_and_save(epochs, trainset_path, finger, focus_id, focus, save_path, batch_size=30):
    fingerprints = Fingerprints(trainset_path, finger, focus_id) 
    fingerprints.pad_with_focus(focus)
    net = SimpleNet(torch.from_numpy(np.array(focus)).float()[None])
    loss_fn = nn.MSELoss()
    optimiser = optim.Adam(net.parameters(), lr=0.01)
    for e in range(epochs):
        for i, (X, y) in enumerate(DataLoader(fingerprints, batch_size=batch_size, shuffle=True)):
            # Preprocessing is done on X as np arrays
            X = torch.from_numpy(np.array(X)).float() # Model uses tensors
            optimiser.zero_grad()
            pred = net(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimiser.step()
            print(f'epoch: {e + 1} batch: {i + 1} loss: {loss.item()}')
    torch.save(net.state_dict(), save_path)

def test(testset_path, finger, focus_id, focus, load_path):
    fingerprints = Fingerprints(testset_path, finger, focus_id)
    fingerprints.pad_with_focus(focus)
    net = SimpleNet(torch.from_numpy(np.array(focus)).float()[None])
    net.load_state_dict(torch.load(load_path))
    batch_size = 30
    net.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(DataLoader(fingerprints, batch_size=batch_size, shuffle=True)):
            X = torch.from_numpy(np.array(X)).float()
            pred = net(X)
            num_correct = sum([1 if(abs(a - b) <= 0.1) else 0 for a, b in zip(pred.detach().numpy(), y.detach().numpy())])
            print(f'batch: {i + 1} accuracy: {(num_correct / batch_size) * 100}%')

DATASET_PATH = 'data/socofing/'
TRAINSET_PATH = DATASET_PATH + 'real'
TESTSET_PATH = lambda difficulty : DATASET_PATH + 'altered/' + difficulty

FINGER = 'left index finger'.split(' ')
FOCUS_ID = random.randint(1, 600)
FOCUS = imgs.get_focus_fingerprint(TRAINSET_PATH, FINGER, FOCUS_ID)

train_and_save(1, TRAINSET_PATH, FINGER, FOCUS_ID, FOCUS, SAVE_PATH)
test(TESTSET_PATH('hard'), FINGER, FOCUS_ID, FOCUS, SAVE_PATH)
# """

"""
import sys

saved = cv2.resize(cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE), (416, 416))[np.newaxis, :, :]
query = cv2.resize(cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE), (416, 416))[np.newaxis, :, :]
net = SimpleNet(torch.from_numpy(np.array(saved)).float()[None])
net.load_state_dict(torch.load(SAVE_PATH))
net.eval()
with torch.no_grad():
    X = torch.from_numpy(np.array(query)).float()[None]
    pred = net(X)
    print(f'pred: {pred.item()}')
"""
