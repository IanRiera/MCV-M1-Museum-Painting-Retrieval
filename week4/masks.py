import os
import cv2 as cv
import numpy as np
from skimage import feature

from random import randrange
import imutils

import scipy.stats as stats
import week4.evaluation as evaluation

import week4.utils as utils

def get_painting_from_mask(img, bg_mask, painting_coords):
    """
    get_painting_from_mask()

    Function to extract a painting from an image using the background mask ...
    """

    # Coordinates of the painting
    tlx = painting_coords[0]
    tly = painting_coords[1]
    brx = painting_coords[2]
    bry = painting_coords[3]

    bg_mask_painting = np.zeros(img.shape, dtype=np.uint8)
    bg_mask_painting[tly:bry, tlx:brx] = 255

    combined = np.array(img.shape, dtype=img.dtype)

    # Combine the image and the background mask
    combined = cv.bitwise_and(img, bg_mask_painting)

    gray_combined = cv.cvtColor(combined,cv.COLOR_BGR2GRAY)

    # Coordinates of non-black pixels.
    coords = np.argwhere(gray_combined)

    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1

    # Get the contents of the bounding box.
    painting = combined[x0:x1, y0:y1]

    return painting


# -------Get background mask-----------
def get_bg_mask(img, max_paintings, image_id):
    """
    get_bg_mask()

    Function to compute a binary mask of the background using method...
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = feature.canny(gray, sigma=3,low_threshold=20,high_threshold=40)
    mask = np.zeros(gray.shape)

    mask_copy = mask.copy()

    mask_copy[edges] = 255
    mask_copy = cv.convertScaleAbs(mask_copy)

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(30,30))
    closed = cv.morphologyEx(mask_copy, cv.MORPH_CLOSE, kernel)

    cnts = cv.findContours(closed.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:4]

    # loop over the contours from bigger to smaller, and find the biggest one with the right orientation
    for c in cnts:
        # # # approximate to the rectangle
        x, y, w, h = cv.boundingRect(c)
        if w > gray.shape[1]/8 and h > gray.shape[0]/6:
            mask[y:y+h,x:x+w]=255 # fill the mask

    found = False
    mask = cv.convertScaleAbs(mask)
    cnts = cv.findContours(mask.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    paintings_coords_aux = []
    for c in cnts:
          # # approximate to the rectangle
          x, y, w, h = cv.boundingRect(c)
          paintings_coords_aux.append([x,y,x+w,y+h])
          found = True

          if len(paintings_coords_aux) == max_paintings:
              break

    if not found:
        paintings_coords = [0,0,img.shape[1],img.shape[0]]

    else:
        paintings_coords = utils.sort_paintings(paintings_coords_aux)

    return [mask, paintings_coords]

def remove_bg(img, params, image_id):
    [mask, paintings_coords] = get_bg_mask(img, params['remove']['max_paintings'], int(image_id))

    result_bg_path = os.path.join(params['paths']['results'], image_id + '.png')
    cv.imwrite(result_bg_path, mask)

    paintings = []
    for painting_id, painting_coords in enumerate(paintings_coords):
        painting = get_painting_from_mask(img, mask, painting_coords)
        paintings.append(painting)

        result_painting_path = os.path.join(params['paths']['results'],
                                            image_id + '_' + str(painting_id) + '.jpg')
        cv.imwrite(result_painting_path, painting)

    return [paintings, paintings_coords]
