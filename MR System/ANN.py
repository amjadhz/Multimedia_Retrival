from pynndescent import NNDescent
import numpy as np

def construct_kd_tree(data, distance_function):
    index = NNDescent(data, metric=distance_function)
    index.prepare()
    return index

def ann(index: NNDescent, vector: np.ndarray, K):
    inputPoint = np.array(vector)
    results = index.query([inputPoint], K)
    return results