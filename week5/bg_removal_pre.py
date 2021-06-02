
import cv2 as cv
import numpy as np
from skimage import feature
import imutils
import os
import operator
import math
import pickle
# ##----TEST AREA----

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
    
    
    
    
def get_mask_M7(img):
    
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


    paintings_coords=[]
    previous_angle = None
        
    for rect in clean_rectangles:        
        box = cv.boxPoints(rect)
        box = np.int0(box)
       
        p1=(box[0][0],box[0][1])
        p2=(box[1][0],box[1][1])
        p3=(box[2][0],box[2][1])
        p4=(box[3][0],box[3][1])
        angle,w_rect,h_rect=get_angle(box)

        area_rect=w_rect*h_rect
        if previous_angle is None:
            if(area_rect>0.07*area_img):
                paintings_coords.append([angle,[p1,p2,p3,p4]])
                mask=cv.fillConvexPoly(mask,box,255)
                cv.drawContours(img, [box], 0, (36,255,12), 4)
                previous_angle=angle
        else:
            if(abs(angle-previous_angle)<5 or abs(angle-previous_angle)>175 ) and (area_rect>0.07*area_img):
                paintings_coords.append([angle,[p1,p2,p3,p4]])
                mask=cv.fillConvexPoly(mask,box,255)
                cv.drawContours(img, [box], 0, (36,255,12), 4)
                previous_angle=angle
                
    
    cv.imshow("img",imutils.resize(img,height=500))
    cv.waitKey()

    return mask, paintings_coords

# query_path = 'data/qsd1_w5/00019.jpg'
# img = cv.imread(query_path)
# mask,coords= get_mask_M7(img)
# print("detected values: {} ".format(coords))

query_path = 'data/qsd1_w5'
gt = pickle.load(open(os.path.join(query_path, "frames.pkl"), 'rb'))

for query_filename in sorted(os.listdir(query_path)):
    if query_filename.endswith('.jpg'):
        image_id = int(query_filename.replace('.jpg', ''))
        image_path = os.path.join(query_path, query_filename)
        img = cv.imread(image_path)
        mask, coords=get_mask_M7(img)
        print("detected values: {} ,\n gt : {}".format(coords,gt[image_id]))
        cv.imshow("mask",imutils.resize(mask,height=700))
        cv.waitKey()

