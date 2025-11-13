from pynndescent import NNDescent
import numpy as np
import pickle

def construct_kd_tree(data, distance_function):
    index = NNDescent(data, metric=distance_function)
    index.prepare()
    return index

def ann(index: NNDescent, vector: np.ndarray, K):
    inputPoint = np.array(vector)
    results = index.query([inputPoint], K)
    return results

def save_ann_index(index, path):
    with open(path, "wb") as f:
        pickle.dump(index, f)
    print(f"[ANN] Index saved to {path}")

def load_ann_index(path):
    with open(path, "rb") as f:
        index = pickle.load(f)
    print(f"[ANN] Index loaded from {path}")
    return index