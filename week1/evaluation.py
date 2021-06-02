from collections import namedtuple
import numpy as np
import cv2 as cv
import pickle
import os

def box_iou(boxA, boxB):
    # compute the intersection over union of two boxes

    # Format of the boxes is [tlx, tly, brx, bry], where tl and br
    # indicate top-left and bottom-right corners of the box respectively.

    # determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def mean_iou(query_path, gt_boxes_path, pred_boxes_path):

    # image = cv.imread('/home/oscar/workspace/master/modules/m1/project/Team3/data/qsd1_w2/00005.jpg')

    images_paths = []
    for query_filename in sorted(os.listdir(query_path)):
        if query_filename.endswith('.jpg'):
            images_paths.append(os.path.join(query_path, query_filename))

    image_ids = []

    groundtruth_text_boxes = pickle.load(open(gt_boxes_path, 'rb'))
    predicted_text_boxes = pickle.load(open(pred_boxes_path, 'rb'))

    # Adapt the format to be able to evaluate the results...
    groundtruth_text_boxes_eval = []
    # Groundtruth of qsd1_2 in wrong format --> change it
    if 'qsd1_w2' in gt_boxes_path:
        for img_gt_boxes in groundtruth_text_boxes:
            for gt_box in img_gt_boxes:
                tlx = gt_box[0][0]
                tly = gt_box[0][1]
                brx = gt_box[2][0]
                bry = gt_box[2][1]
                groundtruth_text_boxes_eval.append([tlx,tly,brx,bry])

    elif 'qsd2_w2' in gt_boxes_path:
        for groundtruth_text_boxes_per_image in groundtruth_text_boxes:
            for groundtruth_text_box in groundtruth_text_boxes_per_image:
                groundtruth_text_boxes_eval.append(groundtruth_text_box)

    else:
        groundtruth_text_boxes_eval = groundtruth_text_boxes

    predicted_text_boxes_eval = []
    for img_id, predicted_text_boxes_per_image in enumerate(predicted_text_boxes):
        for predicted_text_box in predicted_text_boxes_per_image:
            predicted_text_boxes_eval.append(predicted_text_box)
            image_ids.append(img_id)

    total_boxes = 0
    mean_iou = 0

    # Compute iou for each gt/predicted bounding box
    for box_idx, gt_box in enumerate(groundtruth_text_boxes_eval):
        pred_box = predicted_text_boxes_eval[box_idx]

        image = cv.imread(images_paths[image_ids[box_idx]])

        # draw the ground-truth bounding box along with the predicted
        # bounding box
        cv.rectangle(image, tuple(gt_box[:2]), tuple(gt_box[2:]), (0, 255, 0), 2)
        cv.rectangle(image, tuple(pred_box[:2]), tuple(pred_box[2:]), (0, 0, 255), 2)

        # compute the intersection over union and display it

        iou = box_iou(gt_box, pred_box)
        mean_iou += iou
        total_boxes += 1
        cv.putText(image, "IoU: {:.4f}".format(iou), (10, 30),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        #
        # show the output image
        # cv.imshow("Image", image)
        # cv.waitKey(0)


    mean_iou = mean_iou/float(total_boxes)

    return mean_iou
