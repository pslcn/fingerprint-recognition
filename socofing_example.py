import numpy as np
import random
import glob
import torch

import fingerprint_model

MODEL_PATH = "models/"
SAVE_PATH = MODEL_PATH + "net.pth"

DATASET_PATH = "data/socofing/"
TRAINSET_PATH = DATASET_PATH + "real"
TESTSET_PATH = lambda difficulty: DATASET_PATH + "altered/" + difficulty


FINGER = "left index finger".split(" ")
FOCUS_ID = random.randint(1, 600)
FOCUS = fingerprint_model.get_focus_fingerprint(TESTSET_PATH("hard"), FINGER, FOCUS_ID)


# train_fingerprints = fingerprint_model.Fingerprints(TRAINSET_PATH, FINGER, FOCUS_ID)
