import cv2 as cv
import numpy as np
from tqdm import tqdm

from skimage import feature

import week5.text_boxes as text_boxes

def get_matches_filtered(matches, th=450):
    matches_filtered = []
    for m in matches:
        # print(m.distance)
        if m.distance >= th:
            matches_filtered.append(m)
    return matches_filtered

def orb_descriptor(img, text_box):
    img_gray = cv.cvtColor(img.copy(),cv.COLOR_BGR2GRAY)
    img_gray = cv.resize(img_gray, (256, 256), interpolation=cv.INTER_AREA)

    orb = cv.ORB_create(scaleFactor=1.1, WTA_K=2, fastThreshold=5)
    # orb = cv.ORB_create()

    if text_box is not None:
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[text_box[1]:text_box[3],text_box[0]:text_box[2]] = 255
        mask = (255-mask)
        mask = cv.resize(mask, (256, 256), interpolation=cv.INTER_AREA)

        cv.imwrite('i.jpg', img_gray)
        cv.imwrite('m.jpg', mask)

        kp,des=orb.detectAndCompute(img_gray, mask=mask) # mask=None --> better results (fix boundingx box)
    else:
        kp,des = orb.detectAndCompute(img_gray, mask=None)

    return (kp,des)

def get_top_matches(matches, params):
    thr_matches = params['orb']['thr_matches']
    top_matches = np.argsort(np.array(matches))[-10:][::-1]
    if matches[top_matches[0]] >= thr_matches:
        return top_matches
    else:
        return None

def match_descriptors(bbdd_kp_des, query_des, params):
    max_ratio_aux = params['orb']['max_ratio']
    max_distance_aux = params['orb']['max_distance']
    max_ratio=float(max_ratio_aux.replace(',', '.'))
    max_distance=float(max_distance_aux.replace(',', '.'))

    bd_kp, bd_des = bbdd_kp_des
    if len(bd_kp) > 0:
        matches = len(feature.match_descriptors(bd_des, query_des,
                                                metric='hamming', max_ratio=max_ratio, max_distance=max_distance, p=1))
    else:
        matches = 0
    return matches

def compute_bbdd_orb_descriptors(bbdd_path):
    im = cv.imread(bbdd_path)
    return orb_descriptor(im, None)
