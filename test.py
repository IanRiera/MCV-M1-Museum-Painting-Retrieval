import os
import math
import imutils
import cv2 as cv

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

if __name__ == "__main__":
    img = cv.imread('/home/oscar/workspace/master/modules/m1/project/Team3/data/qsd1_w5/00025.jpg')
    # img = cv.imread('/home/oscar/workspace/master/modules/m1/project/Team3/data/qsd1_w5/00010.jpg')
    # theta = 24.49

    # rotated_mask = imutils.rotate(img.copy(), angle=180-theta)
    # cv.imwrite('rotated_10.jpg', rotated_mask)

    theta = 24.49
    theta_aux = -theta
    # box = [theta_aux, [[96,178], [388,183], [86,615], [386,617]]]
    box = [[146,409], [448,409], [162,795], [460,790]]
    rot_box = [theta]
    rot_box.append(rotate_coords(theta_aux, box, img.shape[:2]))
    print(rot_box)
