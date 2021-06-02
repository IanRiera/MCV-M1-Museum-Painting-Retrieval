import os
import imutils
import cv2 as cv
import numpy as np

import week1.bg_removal_methods as methods

def detect_text_box2(img):
    g = img.copy()
    g[:,:,0] = g[:,:,2] = 0

    gray_g = cv.cvtColor(g, cv.COLOR_BGR2GRAY)
    gbw = cv.adaptiveThreshold(gray_g,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,11,30)

    # Getting the kernel to be used in Gradient
    filterSize =(9, 3)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, filterSize)


    gradient_g = cv.morphologyEx(gbw, cv.MORPH_GRADIENT, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (30,15))
    close_g = cv.morphologyEx(gbw, cv.MORPH_CLOSE, kernel, iterations=3)


    experiment = cv.bitwise_and(gradient_g,close_g)
    grad_closed =cv.morphologyEx(experiment, cv.MORPH_CLOSE, kernel, iterations=3)


    # # Find contours, highlight text areas, and extract ROIs
    cnts = cv.findContours(grad_closed, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    ROI_number = 0
    tlx = 0
    tly = 0
    brx = 0
    bry = 0

    # for idx,c in enumerate(cnts):
    #     if ROI_number !=1:
    #         area = cv.contourArea(c)
    #         # if area > 1000:
    #         x,y,w,h = cv.boundingRect(c)
    #         mask[y:y+h, x:x+w] = 0
    #         cv.drawContours(mask, cnts, idx, (255, 255, 255), -1)
    #         r = float(cv.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
    #
    #         if r > 0.45 and w > img.shape[1]/8 and h > img.shape[0]/40 and w>h and w*h< (img.shape[0]*img.shape[1])*0.6:
    #             cv.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 3)
    #             # print("tl:", [x,y]," br: ", [x + w, y + h])
    #             tlx = x
    #             tly = y
    #             brx = x + w
    #             bry = y + h
    #             # ROI = img[y:y+h, x:x+w]
    #             # cv.imwrite('ROI_{}.png'.format(ROI_number), ROI)
    #             ROI_number += 1
    #     else:
    #         break# Algorithm to detect bounding box in an image...

    mask = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)

    final_contours_aux = []

    for idx in range(len(cnts)):
        x, y, w, h = cv.boundingRect(cnts[idx])
        mask[y:y+h, x:x+w] = 0
        cv.drawContours(mask, cnts, idx, (255, 255, 255), -1)
        r = float(cv.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

        if r > 0.25 and w > 2*h and w*h < (img.shape[0]*img.shape[1])*0.4 and w < img.shape[1]*0.95:
            final_contours_aux.append([x,y,w,h])

    final_contours = []

    if len(final_contours_aux) > 0:
        max_area = 0
        max_area_idx = 0
        for idx, cnt in enumerate(final_contours_aux):
            if cnt[2]*cnt[3] > max_area:
                max_area = cnt[2]*cnt[3]
                max_area_idx = idx
        final_contours.append(final_contours_aux[max_area_idx])
    else:
        final_contours.append([0,0,0,0])

        final_contours_aux2 = []
        print('aaaa')
        for idx in range(len(cnts)):
            print('eeee')
            x, y, w, h = cv.boundingRect(cnts[idx])
            mask[y:y+h, x:x+w] = 0
            cv.drawContours(mask, cnts, idx, (255, 255, 255), -1)
            r = float(cv.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
            final_contours_aux2.append([x,y,w,h])
        if len(final_contours_aux2) > 0:
            max_area = 0
            max_area_idx = 0
            for idx, cnt in enumerate(final_contours_aux2):
                if cnt[2]*cnt[3] > max_area:
                    max_area = cnt[2]*cnt[3]
                    max_area_idx = idx
            final_contours.append(final_contours_aux2[max_area_idx])

    return [final_contours[0][0], final_contours[0][1], final_contours[0][2], final_contours[0][3]]

def detect_text_box1(image, img_v):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15,7))
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    th = cv.morphologyEx(img_gray, cv.MORPH_TOPHAT, kernel)
    dilate = cv.morphologyEx(th, cv.MORPH_DILATE, kernel)
    close = cv.morphologyEx(dilate, cv.MORPH_CLOSE, kernel)
    grad = cv.morphologyEx(close, cv.MORPH_GRADIENT, kernel)
    otsu, img_thr = cv.threshold(grad, 0, 255, cv.THRESH_OTSU,)

    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv.findContours(img_thr.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    mask = np.zeros([img_v.shape[0], img_v.shape[1]], dtype=np.uint8)

    final_contours_aux = []

    for idx in range(len(contours)):
        x, y, w, h = cv.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        cv.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

        if r > 0.25 and w > img_v.shape[1]/4 and h > img_v.shape[0]/40 and w < img_v.shape[1]*0.95 and h < img_v.shape[0]*0.25 and w > 2*h and w*h < (img_v.shape[0]*img_v.shape[1])*0.4:
            final_contours_aux.append([x,y,w,h])

    final_contours = []

    if len(final_contours_aux) > 0:
        max_area = 0
        max_area_idx = 0
        for idx, contours in enumerate(final_contours_aux):
            if contours[2]*contours[3] > max_area:
                max_area = contours[2]*contours[3]
                max_area_idx = idx

        final_contours.append(final_contours_aux[max_area_idx])
    else:
        final_contours.append(detect_text_box2(image))

    return [final_contours[0][0], final_contours[0][1], final_contours[0][2], final_contours[0][3]]

def detect_text_box(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    laplacian = cv.Laplacian(gray, cv.CV_64F)
    laplacian = cv.convertScaleAbs(laplacian)
    cnt = cv.findContours(laplacian, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    area_imagen = image.shape[0] * image.shape[1]

    xm = 0
    ym = 0
    wm = 0
    hm = 0

    area_max = 0
    for c in cnt[0]:
        x,y,w,h = cv.boundingRect(c)
        area = w*h
        # area = cv.contourArea(c)
        if area > 1000 and ((area / area_imagen) * 100 < 25):
            x, y, w, h = cv.boundingRect(c)
            if area > area_max:
                area_max = area
                xm, ym, wm, hm = x, y, w, h

    if area_max == 0:
        sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
        sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
        sobel = sobely + sobelx
        sobel = cv.convertScaleAbs(sobel)
        sobel = (255 - sobel)
        sobel = cv.GaussianBlur(sobel, (3, 3), 0)
        cnt2 = cv.findContours(sobel, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        area_max = 0
        for c in cnt2[0]:
            x, y, w, h = cv.boundingRect(c)
            area = h * w
            if area > 500 and ((area / area_imagen) * 100) < 20:
                if area > area_max:
                    area_max = area
                    xm, ym, wm, hm = x, y, w, h
    cv.rectangle(image, (xm, ym), (xm + wm, ym + hm), (36, 255, 12), 3)

    return [xm,ym,xm + wm,hm + ym], image


def mask_evaluation(mask_path, groundtruth_path):
    """
    mask_evaluation()

    Function to evaluate a mask at a pixel level...
    """

    mask = cv.imread(mask_path,0) / 255
    mask_vector = mask.reshape(-1)

    groundtruth = cv.imread(groundtruth_path,0) / 255
    groundtruth_vector = groundtruth.reshape(-1)

    tp = np.dot(groundtruth_vector, mask_vector) # true positive
    fp = np.dot(1-groundtruth_vector, mask_vector) # false positive
    tn = np.dot(1-groundtruth_vector, 1-mask_vector) # true negative
    fn = np.dot(groundtruth_vector, 1-mask_vector) # false positive

    epsilon = 1e-10
    precision = tp / (tp+fp+epsilon)
    recall = tp / (tp+fn+epsilon)
    f1 = 2 * (precision*recall) / (precision+recall+epsilon)

    return precision, recall, f1

def mask_average_evaluation(masks_path, groundtruth_path):
    """
    mask_average_evaluation()

    Function to evaluate all masks at a pixel level...
    """

    avg_precision = 0
    avg_recall = 0
    avg_f1 = 0
    n_masks = 0

    for mask_filename in sorted(os.listdir(masks_path)):
        if mask_filename.endswith('.png'):
            precision, recall, f1 = mask_evaluation(os.path.join(masks_path, mask_filename),
                                                        os.path.join(groundtruth_path, mask_filename.split(".")[0]+'.png'))
            print(" Mask #{} --> Precision: {:.2f}, Recall: {:.2f}, F1-score: {:.2f}".format(mask_filename.split(".")[0], precision, recall, f1))

            avg_precision += precision
            avg_recall += recall
            avg_f1 += f1
            n_masks += 1

    return avg_precision/n_masks, avg_recall/n_masks, avg_f1/n_masks

def compute_fg(image_path, bg_mask_path, painting_coords, bg_results_path):
    """
    compute_fg()

    Function to compute the foreground of an image using the background mask ...
    """

    # Coordinates of the painting
    tlx = painting_coords[0]
    tly = painting_coords[1]
    brx = painting_coords[2]
    bry = painting_coords[3]

    bg_mask = cv.imread(bg_mask_path)
    bg_mask_painting = np.zeros(bg_mask.shape, dtype=np.uint8)
    bg_mask_painting[tly:bry, tlx:brx] = 255

    # cv.imshow('bg_mask', bg_mask)
    # cv.imshow('bg_mask_painting',bg_mask_painting)
    # cv.waitKey()

    # Combine the image and the background mask
    combined = cv.bitwise_and(cv.imread(image_path), bg_mask_painting)

    gray_combined = cv.cvtColor(combined,cv.COLOR_BGR2GRAY)

    # Coordinates of non-black pixels.
    coords = np.argwhere(gray_combined)

    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1

    # Get the contents of the bounding box.
    painting = combined[x0:x1, y0:y1]

    # Save foreground image
    cv.imwrite(bg_results_path, painting)

    # cv.imshow('bg_mask', bg_mask)
    # cv.imshow('bg_mask_painting', bg_mask_painting)
    # cv.imshow('painting', painting)
    # cv.waitKey()

    return painting

def compute_bg_mask(image_path, bg_mask_path, method="M0", color_space="RGB"):
    """
    compute_bg_mask()

    Function to compute a background mask using an specific method...
    """

    image = cv.imread(image_path)

    if method == "M0":
        mask = methods.get_mask_M0(image)

    elif method == "M1":
        mask = methods.get_mask_M1(image, color_space)

    elif method == "M2":
        mask = methods.get_mask_M2(image, color_space)

    elif method == "M3":
        mask = methods.get_mask_M3(image)

    elif method == "M4":
        [mask, paintings_coords] = methods.get_mask_M4(image)

    elif method == "M5":
        [mask, paintings_coords] = methods.get_mask_M5(image)

    # Save background mask
    cv.imwrite(bg_mask_path, mask)

    return paintings_coords
