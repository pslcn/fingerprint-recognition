import numpy as np
import random
import glob
import cv2
import scipy
import torch

def paths_for_finger(path, finger):
	return glob.glob(path + f"/*{finger[0].capitalize()}_{'_'.join([e for e in finger[1:]])}*.BMP")

def is_img_focus(img_path, focus_id):
	return int((img_path.split("/")[-1]).split("_")[0]) == focus_id

def cv2_load_image_grayscale(img_path, img_dim):
	return cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), img_dim)

def fingerprint_data(path, finger, focus_id, img_dim=(64, 64)):
	fingerprint_paths = paths_for_finger(path, finger)
	fingerprints = np.zeros((len(fingerprint_paths), *img_dim), dtype=np.uint8)
	for f in range(fingerprints.shape[0]):
		fingerprints[f] = cv2_load_image_grayscale(fingerprint_paths[f], img_dim).astype(np.uint8)
	return fingerprints

class FingerprintImgs:
  def __init__(self, path, img_dim=(64, 64), batch_size=32):
    self.fingerprint_paths = glob.glob(path + "/*.BMP")
    self.batch_size, self.img_dim = batch_size, img_dim
    self.shuffle()

  def shuffle(self): np.random.shuffle(self.fingerprint_paths)

  def __len__(self): return len(self.fingerprint_paths)

  def __iter__(self): 
    self.batchidx = 0
    return self

  def load_batch(self, start, batch_size): 
    batch = np.zeros((batch_size, *self.img_dim), dtype=np.float16)
    for i in range(start, start + batch_size):
      batch[i - start] = cv2_load_image_grayscale(self.fingerprint_paths[i], self.img_dim).astype(np.float16)
    return batch, self.fingerprint_paths[start: start + batch_size]

  def __next__(self):
    batch_size = len(self.fingerprint_paths) - (self.batchidx * self.batch_size)
    if batch_size < 0: raise StopIteration
    self.batchidx += 1
    return self.load_batch((self.batchidx - 1) * self.batch_size, self.batch_size if batch_size > self.batch_size else batch_size)

def normalise_imgs(imgs):
  return imgs / 255


def principal_component_analysis(imgs, components_k=100):
	imgs = imgs.reshape((imgs.shape[0], IMG_DIM[0] ** 2))
	imgs -= np.mean(imgs, axis=0)
	cov_mat = np.cov(imgs, rowvar=False)
	eig_val, eig_vec = np.linalg.eig(cov_mat)
	idxs = eig_val.argsort()[::-1]
	eig_val, eig_vec = eig_val[idxs], eig_vec[:, idxs]
	eig_vec_k = eig_vec[:, :components_k]
	return np.dot(eig_vec_k.T, imgs.T).T
