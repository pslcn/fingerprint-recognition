import numpy as np
import random
import glob
import torch
import matplotlib.pyplot as plt

import fingerprint_model

DATASET_PATH = "data/socofing/"
altered_dir_path = lambda difficulty: DATASET_PATH + "altered/" + difficulty

finger = "left index finger".split(" ")
focus_id = random.randint(1, 600)
# focus = fingerprint_model.get_focus_fingerprint(altered_dir_path("hard"), FINGER, FOCUS_ID)

# train_fingerprints = fingerprint_model.fingerprint_data(DATASET_PATH + "real", finger, focus_id, img_dim=(128, 128))
# train_fingerprints = fingerprint_model.normalise_imgs(train_fingerprints)
# print(train_fingerprints.shape, train_fingerprints.dtype)

fingerprints = fingerprint_model.FingerprintImgs(DATASET_PATH + "real", batch_size=32)
# for i, l in iter(fingerprints): print(i.shape, len(l))
first_batch, _ = next(iter(fingerprints))
first_batch = fingerprint_model.normalise_imgs(first_batch)
plt.imshow(first_batch[0], cmap="gray")
plt.show()
