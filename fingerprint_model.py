import numpy as np
import random
import glob
import cv2
import scipy
import torch

import parse_dataset

IMG_DIM = (64, 64)

def cv2_load_image_grayscale(img_path, have_3dims=False):
	return cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), IMG_DIM)

def get_focus_fingerprint(path, finger, focus_id):
	for img_path in parse_dataset.fingerprint_paths(path, finger, focus_id):
		if parse_dataset.img_is_focus(img_path, focus_id):
			return cv2_load_image_grayscale(img_path)

def fingerprint_data(path, finger, focus_id):
	fingerprint_paths = parse_dataset.fingerprint_paths(path, finger, focus_id)
	fingerprints = np.zeros((len(fingerprint_paths), *IMG_DIM), dtype=np.uint8)
	for f in range(fingerprints.shape[0]):
		fingerprints[f] = cv2_load_image_grayscale(fingerprint_paths[f]).astype(np.uint8)
	return fingerprints

def principal_component_analysis(imgs, components_k=100):
	imgs = imgs.reshape((imgs.shape[0], IMG_DIM[0] ** 2))
	imgs -= np.mean(imgs, axis=0)
	cov_mat = np.cov(imgs, rowvar=False)
	eig_val, eig_vec = np.linalg.eig(cov_mat)
	idxs = eig_val.argsort()[::-1]
	eig_val, eig_vec = eig_val[idxs], eig_vec[:, idxs]
	eig_vec_k = eig_vec[:, :components_k]
	return np.dot(eig_vec_k.T, imgs.T).T


class FingerprintData:
	def __init__(self, path, finger, focus_id):
		self.labelled_data = []
		for img_path in parse_dataset.fingerprint_paths(path, finger, focus_id):
			self.labelled_data.append((cv2_load_image_grayscale(img_path), torch.tensor([parse_dataset.img_is_focus(img_path, focus_id)], dtype=torch.float32)))

	def shuffle_dataset(self):
		shuffled_idxs = np.arange(0, self.data.shape[0], dtype=np.int64)
		np.random.shuffle(shuffled_idxs)
		self.data = self.data[shuffled_idxs]
