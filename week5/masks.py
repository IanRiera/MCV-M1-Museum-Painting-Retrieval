import os
import cv2 as cv
import numpy as np
from skimage import feature
import math
import imutils
import operator

import week5.rotation as rotation
import week5.evaluation as evaluation
from week5 import utils as utils

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
def get_bg_mask(img, image_id, max_paintings, rotation_theta):
    """
    get_bg_mask_rotated()

    Function to compute a binary mask of the background using method...
    """

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = feature.canny(gray, sigma=3,low_threshold=20,high_threshold=40)
    mask = np.zeros(gray.shape)

    mask_copy = mask.copy()

    mask_copy[edges] = 255
    mask_copy = cv.convertScaleAbs(mask_copy)

    if rotation_theta is not None:
        rotated_mask = imutils.rotate(mask_copy.copy(), angle=rotation_theta)

    else:
        rotated_mask = mask_copy
    # cv.imshow('a', rotated_mask)
    # cv.waitKey()

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(30,30))
    closed = cv.morphologyEx(rotated_mask, cv.MORPH_CLOSE, kernel)

    cnts = cv.findContours(closed.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:4]

    # loop over the contours from bigger to smaller, and find the biggest one with the right orientation
    for c in cnts:

        # # # approximate to the rectangle
        x, y, w, h = cv.boundingRect(c)
        if w < 0.98*gray.shape[1] and w > gray.shape[1]/8 and h > gray.shape[0]/6:
            mask[y:y+h,x:x+w]=255 # fill the mask

    found = False
    mask = cv.convertScaleAbs(mask)
    cnts = cv.findContours(mask.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    paintings_coords_aux = []
    for c in cnts:

        # # approximate to the rectangle
        x, y, w, h = cv.boundingRect(c)

        mask[y:y+h,x:x+w]=255 # fill the mask

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
    [mask, paintings_coords] = get_bg_mask(img.copy(), int(image_id),
                                           params['augmentation']['max_paintings'],
                                           None)

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

def remove_bg_rotate(img, params, image_id):
    rotation_theta = rotation.get_theta(img.copy())

    if 0 <= rotation_theta <= 90:
        rotation_theta_aux = -rotation_theta

    elif 90 < rotation_theta <= 180:
        rotation_theta_aux = 180 - rotation_theta

    [mask, paintings_coords] = get_bg_mask(img.copy(), int(image_id),
                                           params['augmentation']['max_paintings'],
                                           rotation_theta_aux)

    desrotated_mask = imutils.rotate(mask.copy(), angle=-rotation_theta_aux)

    result_bg_path = os.path.join(params['paths']['results'], image_id + '.png')
    cv.imwrite(result_bg_path, desrotated_mask)

    rotated_img = imutils.rotate(img.copy(), angle=rotation_theta_aux)
    paintings = []
    for painting_id, painting_coords in enumerate(paintings_coords):
        painting = get_painting_from_mask(rotated_img, mask, painting_coords)
        paintings.append(painting)

        result_painting_path = os.path.join(params['paths']['results'],
                                            image_id + '_' + str(painting_id) + '.jpg')
        cv.imwrite(result_painting_path, painting)

    paintings_coords_angle = []
    for painting_coords in paintings_coords:
        tlx,tly,brx,bry = painting_coords

        tl_coords = [tlx, tly]
        tr_coords = [brx, tly]
        br_coords = [brx, bry]
        bl_coords = [tlx, bry]

        coords_aux = [tl_coords, tr_coords, br_coords, bl_coords]

        painting_coords_angle = [rotation_theta]
        painting_coords_angle.append(rotation.rotate_coords(rotation_theta_aux, coords_aux, img.shape[:2]))
        paintings_coords_angle.append(painting_coords_angle)

    return [paintings, paintings_coords, paintings_coords_angle]

# ##----Method 2----

def get_angle(box):
    lower_corners=sorted(box, key=operator.itemgetter(1),reverse=True)
    m=(lower_corners[0][1]-lower_corners[1][1])/(lower_corners[0][0]-lower_corners[1][0])
    angle_in_radians = math.atan(m)
    angle_in_degrees = -math.degrees(angle_in_radians)
    if(angle_in_degrees<0):
        angle_in_degrees+=180
    width=abs(lower_corners[0][0]-lower_corners[1][0])
    height=abs(lower_corners[0][1]-lower_corners[2][1])

    return angle_in_degrees,width,height


def area_rect(e):
    area=e[1][0]*e[1][1]
    return area

def discard_overlapping_rectangles(rectangles):

    if len(rectangles)>1:

        biggest_area_rect = rectangles[0]
        clean_rectangles=[biggest_area_rect]

        cx1 = biggest_area_rect[0][0]
        cy1 = biggest_area_rect[0][1]
        w1 = biggest_area_rect[1][1]
        h1 = biggest_area_rect[1][0]

        for rect in rectangles[1:]:
            # to check they are overlapping
            cx2=rect[0][0]
            cy2=rect[0][1]

            overlapping = ((cx1-w1/2) < cx2 < (cx1+w1/2)) and ((cy1-h1/2) < cy2 < (cy1+h1/2))

            if not overlapping:
                clean_rectangles.append(rect)

    else:
         clean_rectangles=rectangles

    return sorted(clean_rectangles, key = area_rect, reverse = True)




def remove_bg_rotate2(img,params,image_id):

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges1 = feature.canny(gray, sigma=3,low_threshold=18,high_threshold=36)

    mask=np.zeros(gray.shape)
    mask1=mask.copy()
    mask1[edges1]=255
    mask1=cv.convertScaleAbs(mask1)

    # cv.imshow("mask", imutils.resize(mask1,height=500))
    # cv.waitKey()

    # 20,20
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(20,20))
    closed = cv.morphologyEx(mask1, cv.MORPH_CLOSE, kernel)

    # cv.imshow("closed", imutils.resize(closed,height=500))
    # cv.waitKey()

    cnts = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:3]

    w_img=img.shape[1]
    h_img=img.shape[0]
    area_img=w_img*h_img

    rectangles=[]
    for c in cnts:
        rectangles.append(cv.minAreaRect(c))

    clean_rectangles=discard_overlapping_rectangles(rectangles)


    paintings_coords_angle=[]
    previous_angle = None
    image=img.copy()
    for rect in clean_rectangles:
        box = cv.boxPoints(rect)
        box = np.int0(box)

        p1=[box[0][0],box[0][1]]
        p2=[box[1][0],box[1][1]]
        p3=[box[2][0],box[2][1]]
        p4=[box[3][0],box[3][1]]
        angle,w_rect,h_rect=get_angle(box)

        area_rect=w_rect*h_rect
        if previous_angle is None:
            if(area_rect>0.07*area_img):
                paintings_coords_angle.append([angle,[p1,p2,p3,p4]])
                mask=cv.fillConvexPoly(mask,box,255)
                cv.drawContours(image, [box], 0, (36,255,12), 4)
                previous_angle=angle
        else:
            if(abs(angle-previous_angle)<5 or abs(angle-previous_angle)>175 ) and (area_rect>0.07*area_img):
                paintings_coords_angle.append([angle,[p1,p2,p3,p4]])
                mask=cv.fillConvexPoly(mask,box,255)
                cv.drawContours(image, [box], 0, (36,255,12), 4)
                previous_angle=angle

    result_bg_path = os.path.join(params['paths']['results'], image_id + '.png')
    cv.imwrite(result_bg_path, mask)

    paintings,paintings_coords,paintings_coords_angle=utils.extract_rotated_paintings(paintings_coords_angle,img)

    for painting_id,painting in enumerate(paintings):

        result_painting_path = os.path.join(params['paths']['results'],
                                            image_id + '_' + str(painting_id) + '.jpg')
        cv.imwrite(result_painting_path, painting)

    return [paintings, paintings_coords, paintings_coords_angle]
