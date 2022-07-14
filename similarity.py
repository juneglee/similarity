import os
import cv2
import numpy as np
from scipy.spatial import distance

# cosine_similarity
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

# euclidean
def euclidean(x, y):
    return distance.euclidean(x, y)

# hamming distance
def read_img(path):
    if os.path.isfile(path):
        return cv2.imread(path, 0)
    else:
        raise ValueError('Path provide is not a valid file : {}'.format(path))

def img2hash(img):
    '''
    :param img: for binary
    :return: for binary
    '''
    gray = cv2.resize(img, (224, 224))
    avg = gray.mean()
    binary = 1 * (gray > avg)
    return binary

def hamming_distance(a, b):
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    # 같은 공간에서의 (0, 1)로 서로 다른 것들을 비교 후 합산한 값
    distance = (a !=b).sum()
    return distance


