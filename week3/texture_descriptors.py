from skimage import feature
import numpy as np
from scipy.fftpack import dct, idct
import cv2 as cv
import matplotlib.pyplot as plt
import math
from numpy import pi
from numpy import r_
from scipy.stats import itemfreq
import pywt

from skimage import exposure
from skimage.feature import hog

# Number of blocks: 16, distance: Correlation, numPoints: 8, radius: 2 --> 0.68 (qsd1_w1)
def lbp_hist(image, numPoints=8, radius=2, eps=1e-7):

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    lbp = feature.local_binary_pattern(gray, numPoints, radius, method="uniform")

    hist = cv.calcHist([lbp.astype(np.uint8)], [0], None, [numPoints + 2], [0, numPoints + 2])

    return hist

# Number of blocks: 16, distance: Intersection, norm: 'ortho', n_coefs: 5 --> 0.89 (qsd1_w1)
def dct_hist(image, norm='ortho', n_coefs=5):

    def dct2(image, norm):
        return dct(dct(image, axis=0, norm=norm), axis=1, norm=norm)

    def idct2(image, norm):
        return idct(idct(image, axis=0, norm=norm), axis=1, norm=norm)

    def zigzag(input, n_coefs):
        h = 0
        v = 0

        vmin = 0
        hmin = 0

        vmax = input.shape[0]
        hmax = input.shape[1]

        output = []

        while (v < vmax) and (h < hmax) and (len(output) < n_coefs):
            output.append(input[v,h])

            if ((h + v) % 2) == 0:                 # going up

                if (v == vmin):

                    if (h == hmax-1):
                        v = v + 1

                    else:
                        h = h + 1

                elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                    v = v + 1

                elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                    v = v - 1
                    h = h + 1

            else:                                    # going down
                if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                    h = h + 1

                elif (h == hmin):                  # if we got to the first column
                    if (v == vmax -1):
                        h = h + 1

                    else:
                        v = v + 1

                elif ((v < vmax -1) and (h > hmin)):     # all other cases
                    v = v + 1
                    h = h - 1

            if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
                output.append(input[v, h])
                break

        return output


    def inverse_zigzag(input, vmax, hmax):

        h = 0
        v = 0

        hmin = 0
        vmin = 0

        output = np.zeros((vmax, hmax))

        i = 0

        while True:
            if ((h + v) % 2) == 0:                 # going up
                if (v == vmin):
                    #print(1)
                    output[v, h] = input[i]        # if we got to the first line

                    if (h == hmax-1):
                        v = v + 1
                    else:
                        h = h + 1

                    i = i + 1

                elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                    #print(2)
                    output[v, h] = input[i]
                    v = v + 1
                    i = i + 1

                elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                    #print(3)
                    output[v, h] = input[i]
                    v = v - 1
                    h = h + 1
                    i = i + 1

            else:                                    # going down
                if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                    #print(4)
                    output[v, h] = input[i]
                    h = h + 1
                    i = i + 1

                elif (h == hmin):                  # if we got to the first column
                    #print(5)
                    output[v, h] = input[i]
                    if (v == vmax-1):
                        h = h + 1
                    else:
                        v = v + 1
                    i = i + 1

                elif((v < vmax -1) and (h > hmin)):     # all other cases
                    output[v, h] = input[i]
                    v = v + 1
                    h = h - 1
                    i = i + 1

            if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
                #print(7)
                output[v, h] = input[i]
                break

            # print(output)
            # print('h, v, i: {}, {}, {}'.format(h, v, i))
            # print('-------')

        return output

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    dct_img = dct2(gray, norm)

    dct_zigzag = np.asarray(zigzag(dct_img, n_coefs))

    hist = cv.calcHist([dct_zigzag.astype(np.uint16)], [0], None, [8], [0, max(dct_zigzag)])

    hist = hist.astype("float32")

    return hist

# Number of blocks: 16, distance: Hellinger, orientations=8, pixels_per_cell: 16x16, cells_per_block: 1x1 --> 0.49 (qsd1_w1)
def hog_hist(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    hog_coefs = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=False, feature_vector=True)
    hog_coefs *= 256

    hist = cv.calcHist([hog_coefs.astype(np.uint8)], [0], None, [8], [0, 256])

    hist = hist.astype("float32")

    return hist

# Number of blocks: 8, distance: Hellinger, level: 3, n_coefs: 7, wavelet='db1' --> 0.89 (qsd1_w1)
def wavelet_hist(image, level=3, n_coefs=7, wavelet='db1'):
    # print(pywt.wavelist())

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    extracted_coefs = pywt.wavedec2(gray, wavelet=wavelet, level=level)

    first_coef, *level_coefs = extracted_coefs

    wavelet_coefs = []
    wavelet_coefs.append(first_coef)

    for i in range(level):
        (LH, HL, HH) = level_coefs[i]
        wavelet_coefs.append(LH)
        wavelet_coefs.append(HL)
        wavelet_coefs.append(HH)

    wavelet_coefs = wavelet_coefs[:n_coefs]

    hist_concat = None
    for cf in wavelet_coefs:
        max_range = abs(np.amax(cf))+1
        hist = cv.calcHist([cf.astype(np.uint8)], [0], None, [8], [0, max_range])

        if hist_concat is None:
            hist_concat = hist
        else:
            hist_concat = cv.hconcat([hist_concat, hist])

    return hist_concat
