import cv2 as cv
import numpy as np
from tqdm import tqdm

from skimage import feature

import week4.text_boxes as text_boxes

def surf_descriptor(image, threshold=400):
    #Find the SURF keypoints and descriptors of a given image
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    surf = cv.xfeatures2d.SURF_create(threshold, )
    kp, des = surf.detectAndCompute(img_gray, None)

    return (kp, des)

def calculate_distance(matches):
    if matches is not None:
        dist = 0
        for m in matches:
            dist += m.distance
        dist /= len(matches)
    else:
        dist = 10000000000000
    return dist

def get_matches_filtered(matches, th=450):
    matches_filtered = []
    for m in matches:
        # print(m.distance)
        if m.distance >= th:
            matches_filtered.append(m)
    return matches_filtered

def orb_descriptor(img, bounding_rm=True):
    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    img_gray = cv.resize(img_gray, (256, 256), interpolation=cv.INTER_AREA)
    orb = cv.ORB_create(scaleFactor=1.1, WTA_K=2, fastThreshold=5)

    # orb = cv.ORB_create()
    if bounding_rm:
        # text_box = text_boxes.detect_text_box(img)
        # mask = np.zeros(img.shape,np.uint8)
        # mask[text_box[1]:text_box[3],text_box[0]:text_box[2]] = 255
        # mask = (255-mask)
        # mask = cv.resize(mask, (256, 256), interpolation=cv.INTER_AREA)

        kp,des=orb.detectAndCompute(img_gray, mask=None) # mask=None --> better results (fix boundingx box)
    else:
        kp,des = orb.detectAndCompute(img_gray, mask=None)

    return (kp,des)

def get_top_matches(matches):
    top_matches = np.argsort(np.array(matches))[-10:][::-1]
    if matches[top_matches[0]] >= 4:
        return top_matches
    else:
        return None

def match_descriptors(bbdd_kp_des, query_des):
    bd_kp, bd_des = bbdd_kp_des
    if len(bd_kp) > 0:
        matches = len(feature.match_descriptors(bd_des, query_des,
                                                metric='hamming', max_ratio=0.8, max_distance=0.8, p=1))
    else:
        matches = 0
    return matches

def compute_bbdd_orb_descriptors(bbdd_path):
    im = cv.imread(bbdd_path)
    return orb_descriptor(im, False)
