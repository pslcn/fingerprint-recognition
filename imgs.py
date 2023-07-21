import glob
import cv2
import numpy as np
import scipy
import random
import torch

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
