import os
import cv2 as cv
import numpy as np
from math import log2, log10, sqrt
from scipy import stats


import pywt
from skimage.restoration import (denoise_wavelet, estimate_sigma)

# ----------------- Wavelet methods ---------------------

def denoise_image_wavelet1(image):
    sigma_est = estimate_sigma(image, multichannel=True, average_sigmas=True)
    denoised_img = denoise_wavelet(image, multichannel=True, convert2ycbcr=False,
                                     method='VisuShrink', mode='soft',
                                     sigma=sigma_est*0.8, rescale_sigma=True)
    return denoised_img

# wavelet options: 'db1', 'haar', 'bior2.8', ... --> print(pywt.wavelist()) to show them
def denoise_image_wavelet2(image, wavelet='bior2.8', level=3):

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # compute coefficients
    coefs = pywt.wavedec2(gray, wavelet, level=level)

    sigma = estimate_sigma(image, multichannel=True, average_sigmas=True)+3
    threshold = sigma*sqrt(2*log2(image.size))

    aux_coefs = list(map(lambda x: pywt.threshold(x,threshold, mode='soft'), coefs))
    denoised_coefs = []
    for cf in aux_coefs:
        denoised_coefs.append(cf)

    # Ugly code --> We must change it (not now)
    if level == 1:
        denoised_img = pywt.waverec2([denoised_coefs[0], [denoised_coefs[1][0], denoised_coefs[1][1],
                                    denoised_coefs[1][2]]], wavelet)
    elif level == 2:
        denoised_img = pywt.waverec2([denoised_coefs[0], [denoised_coefs[1][0], denoised_coefs[1][1], denoised_coefs[1][2]],
                                     [denoised_coefs[2][0], denoised_coefs[2][1], denoised_coefs[2][2]]], wavelet)
    elif level == 3:
        denoised_img = pywt.waverec2([denoised_coefs[0], [denoised_coefs[1][0], denoised_coefs[1][1], denoised_coefs[1][2]],
                                     [denoised_coefs[2][0], denoised_coefs[2][1], denoised_coefs[2][2]],
                                     [denoised_coefs[3][0], denoised_coefs[3][1], denoised_coefs[3][2]]], wavelet)

    return denoised_img

# --------------------------------------


def readImages(path,extension):
    imagenes = dict()
    for filename in sorted(os.listdir(path)):
        if filename.find(extension) != -1:
            img = cv.imread(os.path.join(path, filename))
            imagenes[filename[0:-4]] = img
    return imagenes

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def denoiseImage(img):
    gaussiana =  cv.medianBlur(img,3)
    psnr = PSNR(img, gaussiana)
    if psnr < 35:
        img = gaussiana

    return img


"""path = "../data/qsd1_w3"
imagenes = readImages(path,".jpg")

for img in imagenes:
    imagenes[img] = denoiseImage(imagenes[img],img)
    #cv2.imshow(img,imagenes[img])
    cv2.imwrite("../data/filtradas/"+img+".jpg",imagenes[img])
cv2.waitKey()"""
