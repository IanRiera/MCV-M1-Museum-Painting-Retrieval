import os
import operator
import cv2 as cv
import numpy as np
import pickle
import ntpath

from tqdm import tqdm
import multiprocessing.dummy as mp
from functools import partial
from itertools import repeat

import time

import week1.metrics as metrics
import week3.texture_descriptors as texture

OPENCV_COLOR_SPACES = {
    "RGB": cv.COLOR_BGR2RGB,
    "GRAY": cv.COLOR_BGR2GRAY,
    "LAB": cv.COLOR_BGR2LAB,
    "YCrCb": cv.COLOR_BGR2YCrCb,
    "HSV" : cv.COLOR_BGR2HSV
}

OPENCV_DISTANCE_METRICS = {
    "Correlation": cv.HISTCMP_CORREL,
    "Chi-squared": cv.HISTCMP_CHISQR,
    "Intersection": cv.HISTCMP_INTERSECT,
    "Hellinger": cv.HISTCMP_BHATTACHARYYA
}

TEXTURE_HISTOGRAM_METHODS = {
    "LBP": texture.lbp_hist,
    "DCT": texture.dct_hist,
    "HOG": texture.hog_hist,
    "WAVELET": texture.wavelet_hist
}

def compute_hist_color(image, n_bins=8, color_space="RGB"):
    """
    compute_color_histogram()

    Function to compute ...
    """

    image = cv.cvtColor(image, OPENCV_COLOR_SPACES[color_space])

    n_channels = 1 if color_space == "GRAY" else image.shape[2]

    hist_channels = list(range(n_channels))
    hist_bins = [n_bins,]*n_channels
    hist_range = [0, 256]*n_channels

    hist = cv.calcHist([image], hist_channels, None, hist_bins,
                       hist_range)

    hist = cv.normalize(hist, hist, alpha=0, beta=1,
                        norm_type=cv.NORM_MINMAX).flatten()  # change histogram range from [0,256] to [0,1]

    return hist

def compute_hist_texture(image, method_texture):
    texture_method = TEXTURE_HISTOGRAM_METHODS[method_texture]
    hist = texture_method(image)
    hist = cv.normalize(hist, hist, alpha=0, beta=1,
                        norm_type=cv.NORM_MINMAX)

    return hist

def compute_histogram_blocks(image, text_box, n_bins, color_space, block_size,color=True):

    """
    compute_histogram_blocks()

    Function to compute ...
    """

    # image = cv.cvtColor(image, OPENCV_COLOR_SPACES[color_space])

    if text_box:
        tlx_init = text_box[0]
        tly_init = text_box[1]
        brx_init = text_box[2]
        bry_init = text_box[3]

    sizeX = image.shape[1]
    sizeY = image.shape[0]

    hist_concat = None

    for i in range(0,block_size):
        for j in range(0, block_size):
            # Image block
            img_cell = image[int(i*sizeY/block_size):int(i*sizeY/block_size) + int(sizeY/block_size) ,int(j*sizeX/block_size):int(j*sizeX/block_size) + int(sizeX/block_size)]

            if not text_box:
                if color:
                    hist = compute_color_histogram(img_cell, n_bins, color_space)
                else:
                    hist = compute_texture_histogram(img_cell)

            # If there's a text bounding box ignore the pixels inside it
            else:
                tlx = tlx_init-int(j*sizeX/block_size)
                tly = tly_init-int(i*sizeY/block_size)
                brx = brx_init-int(j*sizeX/block_size)
                bry = bry_init-int(i*sizeY/block_size)

                img_cell_vector = []

                for x in range(img_cell.shape[1]-1):
                    for y in range(img_cell.shape[0]-1):
                        if not (tlx < x < brx and  tly < y < bry):
                            img_cell_vector.append(img_cell[y,x,:])

                img_cell_vector = np.asarray(img_cell_vector)

                n_channels = 1 if color_space == "GRAY" else image.shape[2]
                # Using 3D histograms --> total_bins = n_bins_per_channel ^ n_channels
                hist=np.zeros(n_bins**n_channels,dtype=np.float32)

                if img_cell_vector.size!=0:
                    img_cell_matrix = np.reshape(img_cell_vector,(img_cell_vector.shape[0],1,-1))
                    
                    if color:
                        hist = compute_color_histogram(img_cell_matrix, n_bins, color_space)
                    else: 
                        hist = compute_texture_histogram(img_cell_matrix)

            if hist_concat is None:
                hist_concat = hist
            else:
                hist_concat = cv.hconcat([hist_concat, hist])

    return hist_concat


def compute_histogram_blocks_color(image, text_box, n_bins, color_space, block_size):
    """
    compute_histogram_blocks()

    Function to compute ...
    """

    # image = cv.cvtColor(image, OPENCV_COLOR_SPACES[color_space])

    if text_box:
        tlx_init = text_box[0]
        tly_init = text_box[1]
        brx_init = text_box[2]
        bry_init = text_box[3]

    sizeX = image.shape[1]
    sizeY = image.shape[0]

    hist_concat = None

    for i in range(0,block_size):
        for j in range(0, block_size):
            # Image block
            img_cell = image[int(i*sizeY/block_size):int(i*sizeY/block_size) + int(sizeY/block_size) ,int(j*sizeX/block_size):int(j*sizeX/block_size) + int(sizeX/block_size)]

            if not text_box:
                hist = compute_hist_color(img_cell, n_bins, color_space)

            # If there's a text bounding box ignore the pixels inside it
            else:
                tlx = tlx_init-int(j*sizeX/block_size)
                tly = tly_init-int(i*sizeY/block_size)
                brx = brx_init-int(j*sizeX/block_size)
                bry = bry_init-int(i*sizeY/block_size)

                img_cell_vector = []

                for x in range(img_cell.shape[1]-1):
                    for y in range(img_cell.shape[0]-1):
                        if not (tlx < x < brx and  tly < y < bry):
                            img_cell_vector.append(img_cell[y,x,:])

                img_cell_vector = np.asarray(img_cell_vector)

                n_channels = 1 if color_space == "GRAY" else image.shape[2]
                # Using 3D histograms --> total_bins = n_bins_per_channel ^ n_channels
                # hist=np.zeros(n_bins**n_channels,dtype=np.float32)

                if img_cell_vector.size!=0:
                    img_cell_matrix = np.reshape(img_cell_vector,(img_cell_vector.shape[0],1,-1))
                    hist = compute_histogram_color(img_cell_matrix, n_bins, color_space)

            if hist_concat is None:
                hist_concat = hist
            else:
                hist_concat = cv.hconcat([hist_concat, hist])

    return hist_concat

def compute_multiresolution_histograms_texture(image_path, method_texture, text_box, n_bins = 8, color_space = "RGB"):
    hist_concat = None
    block_sizes = [1, 4, 8, 16]

    for block_size in block_sizes:
        hist = compute_histogram_blocks_texture(image_path, method_texture, text_box, n_bins, color_space,block_size)
        if hist_concat is None:
            hist_concat = hist
        else:
            hist_concat = cv.hconcat([hist_concat, hist])

    return hist_concat

def compute_multiresolution_histograms_color(image_path, text_box, n_bins = 8, color_space = "RGB"):
    hist_concat = None
    block_sizes = [1, 4, 8, 16]

    for block_size in block_sizes:
        hist = compute_histogram_blocks_color(image_path, text_box, n_bins, color_space,block_size)
        if hist_concat is None:
            hist_concat = hist
        else:
            hist_concat = cv.hconcat([hist_concat, hist])

    return hist_concat

def compute_histogram_texture(image_path, text_box, method, method_texture, n_bins, color_space, block_size):

    image_id = int(image_path.split('/')[-1].replace('.jpg', '').replace('bbdd_', ''))
    image = cv.imread(image_path)

    if method == "M1":
        hist = compute_histogram_blocks_texture(image, method_texture, text_box, n_bins, color_space, block_size)
    else:
        hist = compute_multiresolution_histograms_texture(image, method_texture, text_box, n_bins, color_space)

    return hist

def compute_histogram_color(image_path, text_box, method, n_bins, color_space, block_size):

    image_id = int(image_path.split('/')[-1].replace('.jpg', '').replace('bbdd_', ''))
    image = cv.imread(image_path)

    if method == "M1":
        hist = compute_histogram_blocks(image, text_box, n_bins, color_space, block_size,True)
    else:
        hist = compute_histogram_blocks(image, text_box, n_bins, color_space, block_size, False)

    return hist

def compute_bbdd_histograms_texture(bbdd_path, method="M1", method_texture="DCT", n_bins=8, color_space="RGB", block_size=16):

    #glob
    bbdd_paths = []
    for bbdd_filename in sorted(os.listdir(bbdd_path)):
        if bbdd_filename.endswith('.jpg'):
            bbdd_paths.append(os.path.join(bbdd_path, bbdd_filename))

    compute_histogram_partial = partial(compute_histogram_texture, text_box=None, method=method, method_texture=method_texture,
                                        n_bins=n_bins, color_space=color_space, block_size=block_size)

    processes = 4
    with mp.Pool(processes=processes) as p:
        hists = list(tqdm(p.imap(compute_histogram_partial, [path for path in bbdd_paths]), total=len(bbdd_paths)))

    histograms = {}
    for i,h in enumerate(hists):
        histograms[i] = h

    return histograms

def compute_bbdd_histograms_color(bbdd_path, method="M1", n_bins=8, color_space="RGB", block_size=16):

    #glob
    bbdd_paths = []
    for bbdd_filename in sorted(os.listdir(bbdd_path)):
        if bbdd_filename.endswith('.jpg'):
            bbdd_paths.append(os.path.join(bbdd_path, bbdd_filename))

    compute_histogram_partial = partial(compute_histogram_color, text_box=None, method=method,
                                        n_bins=n_bins, color_space=color_space, block_size=block_size)

    processes = 4
    with mp.Pool(processes=processes) as p:
        hists = list(tqdm(p.imap(compute_histogram_partial, [path for path in bbdd_paths]), total=len(bbdd_paths)))

    histograms = {}
    for i,h in enumerate(hists):
        histograms[i] = h

    return histograms

def get_k_images_texture(painting, method_texture, bbdd_histograms, text_box, method="M1", k="10", n_bins=8, distance_metric="Hellinger", color_space="RGB", block_size=16):

    reverse = True if distance_metric in ("Correlation", "Intersection") else False

    if method == "M1":
        hist = compute_histogram_blocks_texture(painting, method_texture, text_box, n_bins, color_space, block_size)
    else:
        hist = compute_multiresolution_histograms_texture(painting, method_texture, text_box, n_bins, color_space)

    distances = {}

    for bbdd_id, bbdd_hist in bbdd_histograms.items():
        print(bbdd_id)
        distances[bbdd_id] = cv.compareHist(hist, bbdd_hist, OPENCV_DISTANCE_METRICS[distance_metric])
        # distances[bbdd_id] = metrics.chi2_distance(hist, bbdd_hist)

    k_predicted_images = (sorted(distances.items(), key=operator.itemgetter(1), reverse=reverse))[:k]

    return [predicted_image[0] for predicted_image in k_predicted_images], distances


def get_k_images_color(painting, bbdd_histograms, text_box, method="M1", k="10", n_bins=8, distance_metric="Hellinger", color_space="RGB", block_size=16):

    reverse = True if distance_metric in ("Correlation", "Intersection") else False

    if method == "M1":
        hist = compute_histogram_blocks(painting, text_box, n_bins, color_space, block_size,color=True)
    else:
        hist = compute_histogram_blocks(painting, text_box, n_bins, color_space, block_size,color=False)

    distances = {}

    for bbdd_id, bbdd_hist in bbdd_histograms.items():
        distances[bbdd_id] = cv.compareHist(hist, bbdd_hist, OPENCV_DISTANCE_METRICS[distance_metric])
        # distances[bbdd_id] = metrics.chi2_distance(hist, bbdd_hist)

    k_predicted_images = (sorted(distances.items(), key=operator.itemgetter(1), reverse=reverse))[:k]

    return [predicted_image[0] for predicted_image in k_predicted_images],distances
