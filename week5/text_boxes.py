import cv2 as cv
import numpy as np
import imutils
import os

import week5.utils as utils

def compute_score(rect, img):
    img_height = img.shape[0]
    img_width = img.shape[1]
    img_height_center = img_height / 2.0
    img_width_center = img_width / 2.0
    aspect_ratio = 4.0

    rect_x = rect[0]
    rect_y = rect[1]
    rect_width = rect[2]
    rect_height = rect[3]
    rect_width_center = rect_x + rect_width/2.0
    rect_height_center = rect_y + rect_height/2.0
    rect_ratio = float(rect_width)/rect_height

    x_center_score = abs(rect_width_center-img_width_center)
    if rect_height_center <= img_height_center:
        y_center_score = abs(rect_height_center-img_height/5.0)
    else:
        y_center_score = abs(rect_height_center-img_height*4.0/5.0)

    ratio_score = abs(rect_ratio-aspect_ratio)

    symmetry_score = abs((rect_x - img_width/6.0) - ((rect_x + rect_width) - img_width*5.0/6.0))
    symmetry_center_score = abs((rect_x - img_width_center) - ((rect_x + rect_width) - img_width_center))

    x_center_weight = 0.5
    y_center_weight = 0.2
    ratio_weight = 0.3
    symmetry_weight = 0.4
    symmetry_center_weight = 0.25

    distance = x_center_score*x_center_weight + y_center_score*y_center_weight \
               + ratio_score*ratio_weight + symmetry_score*symmetry_weight + symmetry_center_score*symmetry_center_weight

    # print('------------------')
    # print(f'x_c_s: {x_center_score}, y_c_s: {y_center_score}, r_s: {ratio_score}')
    # print(f'symm1: {symmetry_score}, symm2: {symmetry_center_score}')
    # print(f'distance: {distance}')
    # print('------------------')

    return distance

def get_best_rectangle(rectangles, img):
    distances = []
    for rect in rectangles:
        if rect is not None:
            distances.append(compute_score(rect, img))
        else:
            distances.append(100000000000000)

    idx_best_rectangle = np.argsort(distances)[0]
    return rectangles[idx_best_rectangle]

def get_best_box(img, filter1_size_x, filter1_size_y, threshold, filter2_size_x, filter2_size_y):

    # Getting the kernel to be used in Gradient
    filterSize = (filter1_size_x,filter1_size_y)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, filterSize)
    top_hat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
    black_hat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

    TH = threshold
    top_hat[(top_hat[:,:] < TH) ] = 0
    black_hat[(black_hat[:,:] < TH) ] = 0

    filterSize = (filter2_size_x,filter2_size_y)
    kernel_roi = cv.getStructuringElement(cv.MORPH_RECT, filterSize)

    def _compute_morphology_roi(hat_img, kernel_roi, sigma = 0.025):
        hat_img_del_width_left = int(hat_img.shape[1]*sigma)
        hat_img_del_width_right = int(hat_img.shape[1]*(1-sigma))
        hat_img_del_height_top = int(hat_img.shape[0]*sigma)
        hat_img_del_height_bot = int(hat_img.shape[0]*(1-sigma))

        hat_img_roi = hat_img[hat_img_del_height_top:hat_img_del_height_bot, hat_img_del_width_left:hat_img_del_width_right]
        closed_hat_img_roi=cv.morphologyEx(hat_img_roi,cv.MORPH_CLOSE, kernel_roi)
        closed_hat_img = hat_img
        closed_hat_img[hat_img_del_height_top:hat_img_del_height_bot, hat_img_del_width_left:hat_img_del_width_right] = closed_hat_img_roi

        return closed_hat_img


    closed_top_hat = _compute_morphology_roi(top_hat, kernel_roi)
    closed_black_hat = _compute_morphology_roi(black_hat, kernel_roi)

    def _get_constrained_rectangles(img, imgshow, aux):
        contours = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        w_img = img.shape[1]
        h_img = img.shape[0]

        rectangles_aux = []

        for c in contours:
              # # approximate to the rectangle
              x, y, w, h = cv.boundingRect(c)
              r = float(cv.countNonZero(img[y:y+h, x:x+w])) / (w * h)
              if (w_img/10 < w < w_img*0.95) and (h_img/40 < h < h_img/2) and (w > h*2) and r > 0.35:
                  # cv.rectangle(imgshow, (x, y), (x + w, y + h), (0,0,255), 10)
                  rectangles_aux.append((x,y,w,h))


        return rectangles_aux

    rectangles_top = _get_constrained_rectangles(closed_top_hat, img, 0)
    rectangles_black = _get_constrained_rectangles(closed_black_hat, img, 1)

    rectangles = rectangles_top + rectangles_black

    if len(rectangles) == 0:
        return None
    else:
        final_rect = get_best_rectangle(rectangles, img)

    return final_rect

def detect_text_box(img):
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l,a,b = cv.split(lab)

    best_boxes_lab = []

    best_boxes_lab.append(get_best_box(l, 60, 30, 150, 90, 1))
    best_boxes_lab.append(get_best_box(a, 60, 30, 25, 90, 1))
    best_boxes_lab.append(get_best_box(b, 60, 30, 20, 90, 1))

    for best_box_lab in best_boxes_lab:
        if best_box_lab is not None:
            [x,y,w,h] = best_box_lab
            # cv.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 5)

    final_best_box = get_best_rectangle(best_boxes_lab, img)

    if final_best_box is not None:

        [x,y,w,h] = final_best_box

        final_best_box = [x, y, x+w, y+h]

        [tlx,tly,brx,bry] = final_best_box
        # cv.rectangle(img, (tlx, tly), (brx, bry), (0,255,0), 3)

        def _expand_box(img, box, sigma_x=0.005, sigma_y=0.015):
            [tlx,tly,brx,bry] = box
            tlx_expanded = int(tlx*(1-sigma_x))
            tly_expanded = int(tly*(1-sigma_y))
            brx_expanded = int(brx*(1+sigma_x))
            bry_expanded = int(bry*(1+sigma_y))
            # cv.rectangle(img, (tlx_expanded, tly_expanded), (brx_expanded, bry_expanded), (255,0,0), 3)
            expanded_box = [tlx_expanded, tly_expanded, brx_expanded, bry_expanded]
            return expanded_box

        final_best_box = _expand_box(img, final_best_box)

    return final_best_box

def remove_text(paintings, params, image_id):
    text_boxes = []

    for painting_id in range(len(paintings)):
        painting_path = os.path.join(params['paths']['results'], image_id + '_' + str(painting_id) + '.jpg')
        painting = cv.imread(painting_path)

        text_boxes.append(detect_text_box(painting))

    return [paintings, text_boxes]
