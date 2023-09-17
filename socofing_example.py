import numpy as np
import random
import glob
import torch
import matplotlib.pyplot as plt

import fingerprint_model

DATASET_PATH = "data/socofing/"
TRAINSET_PATH = DATASET_PATH + "real"
TESTSET_PATH = lambda difficulty: DATASET_PATH + "altered/" + difficulty

FINGER = "left index finger".split(" ")
FOCUS_ID = random.randint(1, 600)
# FOCUS = fingerprint_model.get_focus_fingerprint(TESTSET_PATH("hard"), FINGER, FOCUS_ID)

def show_fingerprint(img):
	if img.ndim == 3:
		img = img[0]
	plt.imshow(img.astype("uint8"), cmap="gray")
	plt.show()

train_fingerprints = fingerprint_model.fingerprint_data(TRAINSET_PATH, FINGER, FOCUS_ID, img_dim=(128, 128))
print(train_fingerprints.shape)

# components_k = 1024
# pca = fingerprint_model.principal_component_analysis(train_fingerprints.astype(np.float16), components_k=components_k)
# new_dim = (int(components_k ** 0.5), int(components_k ** 0.5))
# for f in range(10):
# 	show_fingerprint(pca[f].reshape((new_dim)))
