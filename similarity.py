import numpy as np
from scipy.spatial import distance

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

def euclidean(x, y):
    return distance.euclidean(x, y)