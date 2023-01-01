import glob
import cv2
import numpy as np
import scipy
import torch
from torch.utils.data import Dataset, DataLoader

PATHGLOB = lambda path, finger, focus_id: glob.glob(path + f'/*{finger[0].capitalize()}_{"_".join([e for e in finger[1:]])}*.BMP') 
IMGISFOCUS = lambda img_path, focus_id: int((img_path.split('/')[-1]).split('_')[0]) == focus_id
IMG_DIM = (416, 416)

def get_focus_fingerprint(path, finger, focus_id):
    for img_path in PATHGLOB(path, finger, focus_id):
        if IMGISFOCUS(img_path, focus_id):
            return cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), IMG_DIM)[np.newaxis, :, :]
    return None

class Fingerprints(Dataset):
    def __init__(self, path, finger, focus_id):
        self.data = [(cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), IMG_DIM)[np.newaxis, :, :], torch.tensor([1]).float() if IMGISFOCUS(img_path, focus_id) else torch.tensor([0]).float()) for img_path in PATHGLOB(path, finger, focus_id)]

    def pad_with_focus(self, focus):
        for o in range(len(self.data) // 4):
            self.data.append((focus, torch.tensor([1]).float()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def morph_op(img):
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype('uint8')
    return cv2.dilate(cv2.erode(img.numpy(), cross, iterations=1), cross, iterations=1)

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

def vector_features_cosdist(v1, v2):
    return scipy.spatial.distance.cdist(v1, v2, 'cosine')

def cv2_kaze_descriptors(img, vector_size=32):
    alg = cv2.KAZE_create()
    kps = alg.detect(img)
    kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
    kps, dsc = alg.compute(img, kps)
    dsc = dsc.flatten()
    needed_size = (vector_size * 64)
    if dsc.size < needed_size:
        dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    return dsc

def cv2_orb_brute_force(saved, query):
    orb = cv2.ORB_create()
    saved_keypoints, saved_descriptor = orb.detectAndCompute(saved, None)
    query_keypoints, query_descriptor = orb.detectAndCompute(query, None)
    keypoints_without_size = np.copy(saved)
    cv2.drawKeypoints(saved, saved_keypoints, keypoints_without_size, color=(0, 0, 255))
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(saved_descriptor, query_descriptor)
    matches = sorted(matches, key=lambda x: x.distance)
    img_showing_matches = cv2.drawMatches(query_img, query_keypoints, saved_img, saved_keypoints, matches, saved_img, flags=2)
    num_matches = np.array([len(matches)])
    return num_matches
