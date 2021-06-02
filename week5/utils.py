import os
import sys
import pickle
import numpy as np
from glob import glob
import imutils

import week5.rotation as rotation
from week5 import masks as masks

def path_to_list(data_path, extension='jpg'):
    path_list = sorted(glob(os.path.join(data_path,'*.'+extension)))
    if not path_list:
        str = '[ERROR] No .' + extension + ' files found in directory ' + data_path
        sys.exit(str)
    return path_list

def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(pickle_path, pickle_file):
    with open(pickle_path, 'wb') as f:
        return pickle.dump(pickle_file, f)

def get_image_id(image_path):
    # image_filename = image_path.split('\\')[-1] # IAN
    image_filename = image_path.split('/')[-1]
    image_id = image_filename.split('.')[0]

    return image_id

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
                output[v, h] = input[i]        # if we got to the first line

                if (h == hmax-1):
                    v = v + 1
                else:
                    h = h + 1

                i = i + 1

            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                output[v, h] = input[i]
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                output[v, h] = input[i]
                v = v - 1
                h = h + 1
                i = i + 1

        else:                                    # going down
            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                output[v, h] = input[i]
                h = h + 1
                i = i + 1

            elif (h == hmin):                  # if we got to the first column
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
            output[v, h] = input[i]
            break

    return output

def sort_paintings(paintings_coords_aux):
    paintings_coords = []

    if len(paintings_coords_aux) == 1:
        paintings_coords.append(paintings_coords_aux[0])

    if len(paintings_coords_aux) == 2:
        tlx1 = paintings_coords_aux[0][0]
        tly1 = paintings_coords_aux[0][1]
        brx1 = paintings_coords_aux[0][2]
        bry1 = paintings_coords_aux[0][3]

        tlx2 = paintings_coords_aux[1][0]
        tly2 = paintings_coords_aux[1][1]
        brx2 = paintings_coords_aux[1][2]
        bry2 = paintings_coords_aux[1][3]

        if (tlx1 < tlx2 and brx1 < tlx2) or (tly1 < tly2 and bry1 < tly2):
            paintings_coords.append(paintings_coords_aux[0])
            paintings_coords.append(paintings_coords_aux[1])
        else:
            paintings_coords.append(paintings_coords_aux[1])
            paintings_coords.append(paintings_coords_aux[0])

    elif len(paintings_coords_aux)==3:
        tlx1 = paintings_coords_aux[0][0]
        tly1 = paintings_coords_aux[0][1]
        brx1 = paintings_coords_aux[0][2]
        bry1 = paintings_coords_aux[0][3]

        tlx2 = paintings_coords_aux[1][0]
        tly2 = paintings_coords_aux[1][1]
        brx2 = paintings_coords_aux[1][2]
        bry2 = paintings_coords_aux[1][3]

        tlx3 = paintings_coords_aux[2][0]
        tly3 = paintings_coords_aux[2][1]
        brx3 = paintings_coords_aux[2][2]
        bry3 = paintings_coords_aux[2][3]

        left_12 = tlx1 < tlx2 and brx1 < tlx2
        left_13 = tlx1 < tlx3 and brx1 < tlx3
        left_23 = tlx2 < tlx3 and brx2 < tlx3
        above_12 = tly1 < tly2 and bry1 < tly2
        above_13 = tly1 < tly3 and bry1 < tly3
        above_23 = tly2 < tly3 and bry2 < tly3

        if (left_12 and left_13) or (above_12 and above_13):
            if left_23 or above_23:
                paintings_coords.append(paintings_coords_aux[0])
                paintings_coords.append(paintings_coords_aux[1])
                paintings_coords.append(paintings_coords_aux[2])
            else:
                paintings_coords.append(paintings_coords_aux[0])
                paintings_coords.append(paintings_coords_aux[2])
                paintings_coords.append(paintings_coords_aux[1])

        elif left_12 or above_12:
            paintings_coords.append(paintings_coords_aux[2])
            paintings_coords.append(paintings_coords_aux[0])
            paintings_coords.append(paintings_coords_aux[1])

        elif left_13 or above_13:
            paintings_coords.append(paintings_coords_aux[1])
            paintings_coords.append(paintings_coords_aux[0])
            paintings_coords.append(paintings_coords_aux[2])

        else:
            if left_23 or above_23:
                paintings_coords.append(paintings_coords_aux[1])
                paintings_coords.append(paintings_coords_aux[2])
                paintings_coords.append(paintings_coords_aux[0])
            else:
                paintings_coords.append(paintings_coords_aux[2])
                paintings_coords.append(paintings_coords_aux[1])
                paintings_coords.append(paintings_coords_aux[0])

    return paintings_coords


## For method 2 of removing rotated bg
def get_tl(p):
    return p[0]+p[1]

def extract_rotated_paintings(paintings_coords,img):
    non_rotated_boxes=[]

    # to unrotate the original non-straight painting coords to a paralel to axis version
    for painting_coord in paintings_coords:
        theta,box=painting_coord
        if 0 <= theta <= 90:
            theta_aux = theta
        elif 90 < theta <= 180:
            theta_aux = -(180 - theta)

        non_rotated_box = rotation.rotate_coords(theta_aux,box,img.shape[:2])
        non_rotated_boxes.append(non_rotated_box)

    # to sort comparing the auxiliar unrotated boxes
    sorted_paintings_coords=[]
    sorted_paintings_coords_angle=[]
    rotated_img = imutils.rotate(img.copy(), angle=-theta_aux)
    sorted_paintings = []

    if len(paintings_coords) == 1:
        box1=non_rotated_boxes[0]
        tl1_coords=sorted(box1, key = get_tl, reverse = False)
        tlx1 = int(tl1_coords[0][0])
        tly1 = int(tl1_coords[0][1])
        brx1 = int(tl1_coords[-1][0])
        bry1 = int(tl1_coords[-1][1])

        sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx1,tly1,brx1,bry1]))
        sorted_paintings_coords.append([tlx1,tly1,brx1,bry1])
        sorted_paintings_coords_angle.append(paintings_coords[0])

    elif len(paintings_coords) == 2:
        box1=non_rotated_boxes[0]
        tl1_coords=sorted(box1, key = get_tl, reverse = False)
        tlx1 = int(tl1_coords[0][0])
        tly1 = int(tl1_coords[0][1])
        brx1 = int(tl1_coords[-1][0])
        bry1 = int(tl1_coords[-1][1])

        box2=non_rotated_boxes[1]
        tl2_coords=sorted(box2, key = get_tl, reverse = False)
        tlx2 = int(tl2_coords[0][0])
        tly2 = int(tl2_coords[0][1])
        brx2 = int(tl2_coords[-1][0])
        bry2 = int(tl2_coords[-1][1])

        if (tlx1 < tlx2 and brx1 < tlx2) or (tly1 < tly2 and bry1 < tly2):
            sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx1,tly1,brx1,bry1]))
            sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx2,tly2,brx2,bry2]))

            sorted_paintings_coords.append([tlx1,tly1,brx1,bry1])
            sorted_paintings_coords.append([tlx2,tly2,brx2,bry2])

            sorted_paintings_coords_angle.append(paintings_coords[0])
            sorted_paintings_coords_angle.append(paintings_coords[1])

        else:
            sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx2,tly2,brx2,bry2]))
            sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx1,tly1,brx1,bry1]))

            sorted_paintings_coords.append([tlx2,tly2,brx2,bry2])
            sorted_paintings_coords.append([tlx1,tly1,brx1,bry1])

            sorted_paintings_coords_angle.append(paintings_coords[1])
            sorted_paintings_coords_angle.append(paintings_coords[0])

    elif len(paintings_coords)==3:
        box1=non_rotated_boxes[0]
        tl1_coords=sorted(box1, key = get_tl, reverse = False)
        tlx1 = int(tl1_coords[0][0])
        tly1 = int(tl1_coords[0][1])
        brx1 = int(tl1_coords[-1][0])
        bry1 = int(tl1_coords[-1][1])

        box2=non_rotated_boxes[1]
        tl2_coords=sorted(box2, key = get_tl, reverse = False)
        tlx2 = int(tl2_coords[0][0])
        tly2 = int(tl2_coords[0][1])
        brx2 = int(tl2_coords[-1][0])
        bry2 = int(tl2_coords[-1][1])

        box3=non_rotated_boxes[2]
        tl3_coords=sorted(box3, key = get_tl, reverse = False)
        tlx3 = int(tl3_coords[0][0])
        tly3 = int(tl3_coords[0][1])
        brx3 = int(tl3_coords[-1][0])
        bry3 = int(tl3_coords[-1][1])

        left_12 = tlx1 < tlx2 and brx1 < tlx2
        left_13 = tlx1 < tlx3 and brx1 < tlx3
        left_23 = tlx2 < tlx3 and brx2 < tlx3
        above_12 = tly1 < tly2 and bry1 < tly2
        above_13 = tly1 < tly3 and bry1 < tly3
        above_23 = tly2 < tly3 and bry2 < tly3

        if (left_12 and left_13) or (above_12 and above_13):
            if left_23 or above_23:
                sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx1,tly1,brx1,bry1]))
                sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx2,tly2,brx2,bry2]))
                sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx3,tly3,brx3,bry3]))

                sorted_paintings_coords.append([tlx1,tly1,brx1,bry1])
                sorted_paintings_coords.append([tlx2,tly2,brx2,bry2])
                sorted_paintings_coords.append([tlx3,tly3,brx3,bry3])

                sorted_paintings_coords_angle.append(paintings_coords[0])
                sorted_paintings_coords_angle.append(paintings_coords[1])
                sorted_paintings_coords_angle.append(paintings_coords[2])
            else:
                sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx1,tly1,brx1,bry1]))
                sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx3,tly3,brx3,bry3]))
                sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx2,tly2,brx2,bry2]))

                sorted_paintings_coords.append([tlx1,tly1,brx1,bry1])
                sorted_paintings_coords.append([tlx3,tly3,brx3,bry3])
                sorted_paintings_coords.append([tlx2,tly2,brx2,bry2])

                sorted_paintings_coords_angle.append(paintings_coords[0])
                sorted_paintings_coords_angle.append(paintings_coords[2])
                sorted_paintings_coords_angle.append(paintings_coords[1])

        elif left_12 or above_12:
            sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx3,tly3,brx3,bry3]))
            sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx1,tly1,brx1,bry1]))
            sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx2,tly2,brx2,bry2]))

            sorted_paintings_coords.append([tlx3,tly3,brx3,bry3])
            sorted_paintings_coords.append([tlx1,tly1,brx1,bry1])
            sorted_paintings_coords.append([tlx2,tly2,brx2,bry2])

            sorted_paintings_coords_angle.append(paintings_coords[2])
            sorted_paintings_coords_angle.append(paintings_coords[0])
            sorted_paintings_coords_angle.append(paintings_coords[1])

        elif left_13 or above_13:
            sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx2,tly2,brx2,bry2]))
            sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx1,tly1,brx1,bry1]))
            sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx3,tly3,brx3,bry3]))

            sorted_paintings_coords.append([tlx2,tly2,brx2,bry2])
            sorted_paintings_coords.append([tlx1,tly1,brx1,bry1])
            sorted_paintings_coords.append([tlx3,tly3,brx3,bry3])

            sorted_paintings_coords_angle.append(paintings_coords[1])
            sorted_paintings_coords_angle.append(paintings_coords[0])
            sorted_paintings_coords_angle.append(paintings_coords[2])

        else:
            if left_23 or above_23:
                sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx2,tly2,brx2,bry2]))
                sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx3,tly3,brx3,bry3]))
                sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx1,tly1,brx1,bry1]))

                sorted_paintings_coords.append([tlx2,tly2,brx2,bry2])
                sorted_paintings_coords.append([tlx3,tly3,brx3,bry3])
                sorted_paintings_coords.append([tlx1,tly1,brx1,bry1])

                sorted_paintings_coords_angle.append(paintings_coords[1])
                sorted_paintings_coords_angle.append(paintings_coords[2])
                sorted_paintings_coords_angle.append(paintings_coords[0])
            else:
                sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx3,tly3,brx3,bry3]))
                sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx2,tly2,brx2,bry2]))
                sorted_paintings.append(masks.get_painting_from_mask(rotated_img, None, [tlx1,tly1,brx1,bry1]))

                sorted_paintings_coords.append([tlx3,tly3,brx3,bry3])
                sorted_paintings_coords.append([tlx2,tly2,brx2,bry2])
                sorted_paintings_coords.append([tlx1,tly1,brx1,bry1])

                sorted_paintings_coords_angle.append(paintings_coords[2])
                sorted_paintings_coords_angle.append(paintings_coords[1])
                sorted_paintings_coords_angle.append(paintings_coords[0])



    return [sorted_paintings,sorted_paintings_coords,sorted_paintings_coords_angle]
