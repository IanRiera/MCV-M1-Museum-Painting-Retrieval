import os
import sys
import imutils
import math
import pickle

import cv2 as cv
import numpy as np
from glob import glob
from skimage import feature

import week5.noise_removal as noise_removal

def rotate_coords(theta, box, img_shape):
    h,w = img_shape
    cx = w // 2
    cy = h // 2
    theta_rad = math.radians(-theta)

    rot_box = []

    for box_corner in box:

        x, y = box_corner

        cos_rad = math.cos(theta_rad)
        sin_rad = math.sin(theta_rad)

        rot_x = cos_rad*x + sin_rad*y + (1-cos_rad)*cx - sin_rad*cy
        rot_y = -sin_rad*x + cos_rad*y + sin_rad*cx + (1-cos_rad)*cy
        rot_box.append([rot_x,rot_y])

    return rot_box

def hough_rotation_theta(img_show, img, rho_res, theta_res,
                         min_lines, thr_init, thr_step):

    def _reject_outliers(values, threshold = 2.):
        hist, bin_edges = np.histogram(values)
        lower_bin_edge = np.argmax(hist)
        upper_bin_edge = np.argmax(hist)+1
        most_common_values = [bin_edges[lower_bin_edge] <= v <= bin_edges[upper_bin_edge] for v in values]
        # print(f'Thetas: {values}')
        # print(f'Most common values: {most_common_values}')
        return values[most_common_values]

    found_rotation_theta = False
    iter = 0
    while not found_rotation_theta:
        lines = cv.HoughLines(img.copy(), rho_res, theta_res * np.pi / 180,
                              thr_init-thr_step*iter, None, 0, 0)
        if lines is not None:
            horizontal_lines_thetas = []
            horizontal_lines = []
            vertical_lines = []
            for line in lines:
                theta = line[0][1]
                corrected_theta = 90 - theta * 180 / np.pi
                if corrected_theta < 0:
                    corrected_theta += 180
                if (0 <= corrected_theta <= 45) or (135 <= corrected_theta <= 180):
                    horizontal_lines_thetas.append(corrected_theta)
                    horizontal_lines.append(line)
                else:
                    vertical_lines.append(line)

            if len(horizontal_lines_thetas) >= min_lines:
                found_rotation_theta = True
                inlier_lines_thetas = _reject_outliers(np.array(horizontal_lines_thetas))
                # print(inlier_lines_thetas)
                rotation_theta = sum(inlier_lines_thetas)/len(inlier_lines_thetas)
                # print(rotation_theta)

        iter += 1
    return rotation_theta

def get_theta(img):

    gray = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
    edges = feature.canny(gray, sigma=3,low_threshold=20,high_threshold=40)

    img_edges = np.zeros(gray.shape)
    img_edges[edges] = 255
    img_edges = cv.convertScaleAbs(img_edges)

    rotation_theta = hough_rotation_theta(img, img_edges.copy(), rho_res=1, theta_res=1,
                                          min_lines=3, thr_init=700, thr_step=25)
    return rotation_theta
