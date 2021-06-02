import numpy as np

def euclidean_distance(hist_A, hist_B):
    d = np.sum([(a - b) ** 2
        for (a, b) in zip(hist_A, hist_B)])

    return np.sqrt(d)

def l1_distance(hist_A, hist_B):
    d = np.sum([abs(a - b)
        for (a, b) in zip(hist_A, hist_B)])

    return d

def chi2_distance(hist_A, hist_B, epsilon = 1e-10):
    d = np.sum([((a - b) ** 2) / (a + b + epsilon)
        for (a, b) in zip(hist_A, hist_B)])

    return d

def hellinger_kernel_distance(hist_A, hist_B):
    d = np.sum([np.sqrt(a*b)
        for (a, b) in zip(hist_A, hist_B)])

    return d
