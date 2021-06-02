import os
import sys
import imutils
import math
import argparse
import pickle

import cv2 as cv
import numpy as np
from glob import glob
from skimage import feature

from week5 import utils as utils
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

        # print(rot_x,rot_y)
        rot_box.append([rot_x,rot_y])
    return rot_box

def show_hough_lines(img, lines):
    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv.line(img, pt1, pt2, (0,255,0), 5, cv.LINE_AA)


def hough_rotation_theta(img_show, img, rho_res, theta_res, min_lines,
                        thr_init, thr_step, show_h_lines, show_v_lines):

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
                if show_h_lines:
                    show_hough_lines(img_show, horizontal_lines)
                if show_v_lines:
                    show_hough_lines(img_show, vertical_lines)

        iter += 1
    return rotation_theta

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Find sub-optimal parameters to get rotation theta')

    parser.add_argument('--rho_res', '-r', type=int, default=1)

    parser.add_argument('--theta_res', '-t', type=int, default=1)

    parser.add_argument('--min_lines', '-m', type=int, default=3)

    parser.add_argument('--thr_init', '-i', type=int, default=450)

    parser.add_argument('--thr_step', '-s', type=int, default=25)

    parser.add_argument('--show_h_lines', action='store_true')

    parser.add_argument('--show_v_lines', action='store_true')

    args = parser.parse_args(args)

    return args

if __name__ == "__main__":

    args = parse_args()
    print(args)

    max_paintings = 3

    query_path = '/home/oscar/workspace/master/modules/m1/project/Team3/data/qsd1_w5'
    query_list = sorted(glob(os.path.join(query_path, '*.jpg')))

    gt = pickle.load(open('data/qsd1_w5/frames.pkl','rb'))
    avg_angular_error = 0.0
    count=0

    for image_id, img_path in enumerate(query_list):
        if image_id == 25:
            img = cv.imread(img_path)

            # img,_,_ = noise_removal.denoise_painting(img)

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            cv.imwrite('gray.jpg', gray)
            edges = feature.canny(gray, sigma=3,low_threshold=20,high_threshold=40)

            mask = np.zeros(gray.shape)

            mask_edges = mask.copy()

            mask_edges[edges] = 255
            mask_edges = cv.convertScaleAbs(mask_edges)

            gray_copy = np.copy(gray)
            mask_copy2 = np.copy(mask_edges)

            rotation_theta = hough_rotation_theta(img, mask_copy2, rho_res=args.rho_res, theta_res=args.theta_res,
                                                  min_lines=args.min_lines, thr_init=args.thr_init, thr_step=args.thr_step,
                                                  show_h_lines=args.show_h_lines, show_v_lines=args.show_v_lines)

            cv.imshow('Image',  imutils.resize(img, height=600))
            cv.imwrite('hough_img.jpg', img)

            if 0 <= rotation_theta <= 90:
                rotation_theta_aux = -rotation_theta
            elif 90 < rotation_theta <= 180:
                rotation_theta_aux = 180 - rotation_theta

            cv.imshow('edges', imutils.resize(mask_edges, height=600))
            cv.imwrite('edges.jpg', mask_edges)
            rotated_mask = imutils.rotate(mask_edges, angle=rotation_theta_aux)
            cv.imshow('Rotated mask', imutils.resize(rotated_mask, height=600))

            kernel = cv.getStructuringElement(cv.MORPH_RECT,(30,30))
            closed = cv.morphologyEx(rotated_mask, cv.MORPH_CLOSE, kernel)

            cnts = cv.findContours(closed.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:4]

            for c in cnts:
                x, y, w, h = cv.boundingRect(c)
                if w > gray.shape[1]/8 and h > gray.shape[0]/6:
                    cv.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 10)
                    mask[y:y+h,x:x+w]=255 # fill the mask

            found_painting = False
            mask = cv.convertScaleAbs(mask)
            cnts = cv.findContours(mask.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            # cv.imshow('gray'+str(image_id), imutils.resize(gray, height=600))
            # cv.imshow('rotated_mask'+str(image_id), imutils.resize(rotated_mask, height=600))
            # cv.imshow('closed'+str(image_id), imutils.resize(closed, height=600))
            # cv.imshow('img'+str(image_id), imutils.resize(img, height=600))
            # cv.imshow('mask'+str(image_id), imutils.resize(mask, height=600))

            desrotated_mask = imutils.rotate(mask.copy(), angle=-rotation_theta_aux)
            cv.imshow('Mask', imutils.resize(desrotated_mask, height=600))
            cv.waitKey()

            paintings_coords_aux = []
            for c in cnts:
                  # # approximate to the rectangle
                  x, y, w, h = cv.boundingRect(c)
                  paintings_coords_aux.append([x,y,x+w,y+h])
                  found_painting = True

                  if len(paintings_coords_aux) == max_paintings:
                      break

            if not found_painting:
                paintings_coords = [0,0,img.shape[1],img.shape[0]]

            else:
                paintings_coords = utils.sort_paintings(paintings_coords_aux)

            paintings_coords_angle = []
            for painting_coords in paintings_coords:
                tlx,tly,brx,bry = painting_coords

                tl_coords = [tlx, tly]
                tr_coords = [brx, tly]
                br_coords = [brx, bry]
                bl_coords = [tlx, bry]
                coords_aux = [tl_coords, tr_coords, br_coords, bl_coords]

                painting_coords_angle = [rotation_theta]
                painting_coords_angle.append(rotate_coords(rotation_theta_aux, coords_aux, img.shape[:2]))
                paintings_coords_angle.append(painting_coords_angle)

            # print(paintings_coords_angle)

            # Evaluation
            gt_angles = [x[0] for x in gt[image_id]]
            hy_angles = [l[0] for l in paintings_coords_angle]

            common_vals = min(len(gt_angles), len(hy_angles))
            for kk in range(common_vals):
                gta = gt_angles[kk] * np.pi / 180
                hya = hy_angles[kk] * np.pi / 180

                v1 = [abs(np.cos(gta)),np.sin(gta)]
                v2 = [abs(np.cos(hya)),np.sin(hya)]
                ang_error = abs(np.arccos(np.dot(v1,v2)) * 180 / np.pi)
                avg_angular_error += ang_error

                #avg_angular_error += abs(gt_angles[kk] - hy_angles[kk])
                count = count + 1

                print(f'Img ID: {image_id} -> Err: {ang_error:.2f} Gt: {gt_angles[kk]:.2f} Pred: {hy_angles[kk]:.2f}')

            avg_angular_error /= count
            print(f'Avg error: {avg_angular_error:.2f}')
            print('-------------------------------------------------------------------------------------')
