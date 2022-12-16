import os
import imageio.v2 as imageio

DATASET_URL = 'data/socofing/real'

def get_img(name):
    return imageio.imread(os.path.join(DATASET_URL, name), as_gray=True)

fp = get_img('100__M_Left_index_finger.BMP')

print(fp)
