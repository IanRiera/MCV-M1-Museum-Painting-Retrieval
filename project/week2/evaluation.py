from collections import namedtuple
import numpy as np
import cv2 as cv

Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

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


# TODO: size(all_gt_boxes) != size(all_pred_boxes)
def mean_iou(images_paths, all_gt_boxes, all_pred_boxes):
    total_boxes = 0
    mean_iou = 0

    # Get gt bounding boxes of each image
    for img_idx, img_gt_boxes in enumerate(all_gt_boxes):
        img_pred_boxes = all_pred_boxes[img_idx]
        image = cv.imread(images_paths[img_idx])

        # Compute iou for each gt/predicted bounding box --> Must be changed (todo)
        for box_idx, gt_box in enumerate(img_gt_boxes):

            pred_box = img_pred_boxes[box_idx]

            # draw the ground-truth bounding box along with the predicted
            # bounding box
            cv.rectangle(image, tuple(gt_box[:2]), tuple(gt_box[2:]), (0, 255, 0), 2)
            cv.rectangle(image, tuple(pred_box[:2]), tuple(pred_box[2:]), (0, 0, 255), 2)

            # compute the intersection over union and display it
            iou = box_iou(gt_box, pred_box)
            mean_iou += iou

            cv.putText(image, "IoU: {:.4f}".format(iou), (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            print(iou)

            # show the output image
            cv.imshow("Image", image)
            cv.waitKey(0)

            total_boxes += 1

    mean_iou = mean_iou/float(total_boxes)

    print("{:.4f}".format(mean_iou))

    return mean_iou

# Example (from left to right): 1 gt/pred box, 1 gt/pred box, 2 gt/pred boxes, 2 gt and 1 pred box
gt_boxes =   [[[39,63,203,112]],  [[49,75,203,125]],  [[349,84,498,133],[54,79,217,139]],  [[37,59,197,113],[324,69,483,126]]]
pred_boxes = [[[54,66,198,114]],  [[42,78,186,126]],  [[353,89,501,138],[51,83,209,129]],  [[45,66,201,116]]]

path_to_examples = "/home/oscar/Desktop/"

images_paths = [path_to_examples + "example1.jpg", path_to_examples + "example2.jpg",
                path_to_examples + "example3.jpg", path_to_examples + "example4.jpg"]

mean_iou(images_paths, gt_boxes, pred_boxes)
