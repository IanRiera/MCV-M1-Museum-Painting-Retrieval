import os
import cv2 as cv
import numpy as np
from math import log10, sqrt

def PSNR(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    psnr = 20 * log10(original.max() / sqrt(mse+1e-7))
    return psnr

def denoise_painting(img):
    median =  cv.medianBlur(img,3)
    psnr = PSNR(img, median)
    denoised = False
    if psnr < 31.7:
        img = median
        denoised = True
    return [img, denoised, psnr]

def denoise_paintings(paintings, params, image_id):
    denoised_paintings = []
    for painting_id, painting in enumerate(paintings):
        [painting_denoised, denoised, psnr] = denoise_painting(painting)
        denoised_paintings.append(painting_denoised)

        if denoised:
            # result_painting_path = os.path.join(params['paths']['results'],
            #                                     image_id + '_' + str(painting_id) + '_denoised.jpg')
            result_painting_path = os.path.join(params['paths']['results'],
                                                image_id + '_' + str(painting_id) + '.jpg')
            cv.imwrite(result_painting_path, painting_denoised)

    return denoised_paintings
