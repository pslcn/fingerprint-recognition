import numpy as np
import random
import glob
import cv2
import scipy
import torch

import parse_dataset

IMG_DIM = (416, 416)

def cv2_load_image_grayscale(img_path):
	return cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), IMG_DIM)[np.newaxis, :, :]

def get_focus_fingerprints(path, finger, focus_id):
	focus_fingerprints = []
	for img_path in parse_dataset.fingerprint_paths(path, finger, focus_id):
		if parse_dataset.img_is_focus(img_path, focus_id):
			focus_fingerprints.append(cv2_load_image_grayscale(img_path))
	return focus_fingerprints

class FingerprintsData:
	def __init__(self, path, finger, focus_id):
		self.labelled_data = []
		for img_path in parse_dataset.fingerprint_paths(path, finger, focus_id):
			self.labelled_data.append((cv2_load_image_grayscale(img_path), torch.tensor([parse_dataset.img_is_focus(img_path, focus_id)], dtype=torch.float64)))

	def pad_with_focus(self, focus):
		for o in range(len(self.labelled_data) // 4):
			self.labelled_data.append((focus[random.randint(0, len(focus) - 1)], torch.tensor([1], dtype=torch.float64)))


MODEL_PATH = "models/"
SAVE_PATH = MODEL_PATH + "net.pth"

DATASET_PATH = "data/socofing/"
TRAINSET_PATH = DATASET_PATH + "real"
TESTSET_PATH = lambda difficulty: DATASET_PATH + "altered/" + difficulty


def show_fingerprint(img):
	imgplot = plt.imshow(img.astype("uint8"), cmap="gray")
	plt.show()


# FINGER = "left index finger".split(" ")
# FOCUS_ID = random.randint(1, 600)
# FOCUS = get_focus_fingerprints(TESTSET_PATH("hard"), FINGER, FOCUS_ID)


# train_fingerprints = Fingerprints(TRAINSET_PATH, FINGER, FOCUS_ID)
