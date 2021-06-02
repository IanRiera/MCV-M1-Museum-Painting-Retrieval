# boxes = pickle.load(open(os.path.join(boxes_path), 'rb'))[image_id][0]

from collections import namedtuple
import numpy as np
import cv2 as cv
import pickle
import os
from sklearn.metrics import jaccard_score
import ml_metrics as mlm

from week5 import utils as utils

# -------------------------------------------
# print(f'Img ID: {image_id} -> Err: {ang_error:.2f} Gt: {gt_angles[kk]:.2f} Pred: {hy_angles[kk]:.2f}')

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

def evaluate_bg(bg_predicted_list, bg_groundtruth_list, verbose):
    """
    mask_average_evaluation()

    Function to evaluate all masks at a pixel level...
    """

    avg_precision = 0
    avg_recall = 0
    avg_f1 = 0
    n_masks = 0

    for id, bg_predicted in enumerate(bg_predicted_list):
        precision, recall, f1 = mask_evaluation(bg_predicted, bg_groundtruth_list[id])

        if verbose:
            print(" Mask #{} --> Precision: {:.2f}, Recall: {:.2f}, F1-score: {:.2f}".format(id, precision, recall, f1))

        avg_precision += precision
        avg_recall += recall
        avg_f1 += f1
        n_masks += 1

    return avg_precision/n_masks, avg_recall/n_masks, avg_f1/n_masks

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


def evaluate_text_boxes(gt_boxes_path, pred_boxes_path):
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

    # elif 'qsd2_w2' in gt_boxes_path:
    else:
        for groundtruth_text_boxes_per_image in groundtruth_text_boxes:
            for groundtruth_text_box in groundtruth_text_boxes_per_image:
                groundtruth_text_boxes_eval.append(groundtruth_text_box)

    # else:
    #     groundtruth_text_boxes_eval = groundtruth_text_boxes

    predicted_text_boxes_eval = []
    for img_id, predicted_text_boxes_per_image in enumerate(predicted_text_boxes):
        for predicted_text_box in predicted_text_boxes_per_image:
            predicted_text_boxes_eval.append(predicted_text_box)

    # print(f'Lenght: {len(groundtruth_text_boxes_eval)} --> GROUNDTRUTH: {groundtruth_text_boxes_eval}')
    # print(f'Lenght: {len(predicted_text_boxes_eval)} --> PREDICTED: {predicted_text_boxes_eval}')

    total_boxes = 0
    mean_iou = 0

    # Compute iou for each gt/predicted bounding box
    for box_idx, gt_box in enumerate(groundtruth_text_boxes_eval):
        pred_box = predicted_text_boxes_eval[box_idx]
        # print('----------------')
        # print(f'ID: {box_idx} --> {gt_box}, {pred_box}')

        if pred_box is None:
            iou = 0
        else:
            # compute the intersection over union and display it
            iou = box_iou(gt_box, pred_box)
        mean_iou += iou
        total_boxes += 1

    mean_iou = mean_iou/float(total_boxes)

    return mean_iou

def output_predicted_paintings_test(params, paintings_predicted_list, k):
    predicted_frames = utils.load_pickle(os.path.join(params['paths']['results'], 'frames.pkl'))
    query_list = params['lists']['query']
    for query_image_path in query_list:
        image_id = utils.get_image_id(query_image_path)

        paintings_predicted_image = paintings_predicted_list[int(image_id)]
        frames_coords_image = predicted_frames[int(image_id)]

        print(f'Image: {query_image_path} --> Rotation: {frames_coords_image[0][0]} degrees')

        for painting_id, painting_predicted in enumerate(paintings_predicted_image):
            print(f'-> Painting #{painting_id}')
            print(f'        {k} most similar images: {paintings_predicted_image[painting_id]}')

        print('----------------------')

def output_predicted_paintings(query_list, paintings_predicted_list, paintings_groundtruth_list, k):
    for query_image_path in query_list:
        image_id = utils.get_image_id(query_image_path)

        paintings_predicted_image = paintings_predicted_list[int(image_id)]
        paintings_groundtruth_image = paintings_groundtruth_list[int(image_id)]

        print(f'Image: {query_image_path}')

        for painting_id, painting_groundtruth in enumerate(paintings_groundtruth_image):
            print(f'-> Painting #{painting_id}')
            print(f'    Groundtruth: {painting_groundtruth}')

            # If we detected the painting
            if len(paintings_predicted_image) > painting_id:
                print(f'        {k} most similar images: {paintings_predicted_image[painting_id]}')
            else:
                print('        Painting not detected!!!')

        print('----------------------')

def evaluate_frames(gt_frames_list, predicted_frames_list, params):

    def _create_masks (ima_name, boxes):
        img = cv.imread(ima_name) # trainImage

        height, width, depth = img.shape
        mask = np.zeros((height,width), np.uint8)

        boxes = [np.array(box, np.int32).reshape((-1,1,2)) for box in boxes]
        line_type = 8
        cv.fillPoly(mask, boxes, (1))

        return mask

    images_list = params['lists']['query']

    if len(predicted_frames_list) != len(gt_frames_list) or len(predicted_frames_list) != len(images_list):
        print ('Error: length of frame boxes positions does not match ground truth')
        print (len(predicted_frames_list), len(gt_frames_list), len(images_list))
        sys.exit()

    avg_pic_iou       = 0.0
    avg_angular_error = 0.0
    count = 0

    for jj, ima_name in enumerate(images_list):
        gt_frames    = [x[1] for x in gt_frames_list[jj]]
        hy_frames    = [x[1] for x in predicted_frames_list[jj]]
        gt_mask  = _create_masks(ima_name, gt_frames)
        hy_mask  = _create_masks(ima_name, hy_frames)

        avg_pic_iou       += jaccard_score(gt_mask.flatten(), hy_mask.flatten())

        gt_angles    = [x[0] for x in gt_frames_list[jj]]
        hy_angles    = [x[0] for x in predicted_frames_list[jj]]
        common_vals = min(len(gt_angles), len(hy_angles))
        for kk in range(common_vals):
            gta = gt_angles[kk] * np.pi / 180
            hya = hy_angles[kk] * np.pi / 180

            v1 = [abs(np.cos(gta)),np.sin(gta)]
            v2 = [abs(np.cos(hya)),np.sin(hya)]
            ang_error = abs(np.arccos(np.dot(v1,v2)) * 180 / np.pi)
            avg_angular_error += ang_error

            count = count + 1

    avg_pic_iou       /= len(predicted_frames_list)
    avg_angular_error /= count
    print ('F,{:.3f}, {:.3f}'.format(avg_angular_error, avg_pic_iou))

def evaluate(params, k_list, verbose=False):
    if params['augmentation'] is not None:
        if params['augmentation']['bg']:
            bg_predicted_list = utils.path_to_list(params['paths']['results'], extension='png')
            bg_groundtruth_list = utils.path_to_list(params['paths']['query'], extension='png')
            # assert len(bg_groundtruth_list) == len(bg_predicted_list)
            avg_precision, avg_recall, avg_f1 = evaluate_bg(bg_predicted_list, bg_groundtruth_list, verbose)

            print('**********************')
            print('Average --> Precision: {:.2f}, Recall: {:.2f}, F1-score: {:.2f}'.format(avg_precision, avg_recall, avg_f1))

            if params['augmentation']['rotated']:
                # evaluate frames.pkl
                predicted_frames = utils.load_pickle(os.path.join(params['paths']['results'], 'frames.pkl'))
                gt_frames = utils.load_pickle(os.path.join(params['paths']['query'], 'frames.pkl'))
                evaluate_frames(gt_frames, predicted_frames, params)

        # if params['augmentation'].text_extract:
        #     text_extract_predicted_list = path_to_list(params['paths'].results, extension='txt')
        #     text_extract_groundtruth_list = path_to_list(params['paths'].query, extension='txt')
        #     # assert len(text_groundtruth_list) == len(text_predicted_list)
        #     evaluate_text_extract(text_extract_predicted_list, text_extract_groundtruth_list)

        # if params['augmentation']['text']:
        #     # text_boxes_predicted_list = utils.load_pickle(os.path.join(params['paths']['results'], 'text_boxes.pkl'))
        #     # text_boxes_groundtruth_list = utils.load_pickle(os.path.join(params['paths']['query'], 'text_boxes.pkl'))
        #
        #     mean_iou = evaluate_text_boxes(os.path.join(params['paths']['query'], 'text_boxes.pkl'), os.path.join(params['paths']['results'], 'text_boxes.pkl'))
        #     print('**********************')
        #     print(f'Text bounding boxes evaluation: Mean IOU = {mean_iou}')
        #     print('**********************')

    paintings_predicted_list = utils.load_pickle(os.path.join(params['paths']['results'], 'result.pkl'))
    paintings_groundtruth_list = utils.load_pickle(os.path.join(params['paths']['query'], 'gt_corresps.pkl'))
    if verbose:
        output_predicted_paintings(params['lists']['query'], paintings_predicted_list, paintings_groundtruth_list, max(k_list))

    for k in k_list:

        # Adapt the format of the "gt_corresps.pkl" to be able to evaluate the results...
        groundtruth_paintings_list_eval = []
        predicted_paintings_list_eval = []

        if 'qsd2_w1' in params['paths']['query']:
            groundtruth_paintings_list_eval = paintings_groundtruth_list

            for predicted_paintings_per_image in paintings_predicted_list:
                for predicted_paintings_per_painting in predicted_paintings_per_image:
                    predicted_paintings_list_eval.append(predicted_paintings_per_painting)

        else:
            for image_id, groundtruth_paintings_per_image in enumerate(paintings_groundtruth_list):
                predicted_paintings_per_image = paintings_predicted_list[image_id]
                for painting_id, groundtruth_painting in enumerate(groundtruth_paintings_per_image):
                    groundtruth_paintings_list_eval.append([groundtruth_painting])
                    if len(predicted_paintings_per_image) > painting_id:
                        predicted_painting = predicted_paintings_per_image[painting_id]
                    else:
                        predicted_painting = [-1]

                    predicted_paintings_list_eval.append(predicted_painting)

        # with open("optimize_orb_params.txt", "a") as myfile:
        #     myfile.write('\n')
        #     myfile.write('Params (m, r, d): ' + str(params['orb']['thr_matches']) + ',' + str(params['orb']['max_ratio']) +
        #                  ', ' + str(params['orb']['max_distance']) + ' ----> Map@1 = ' +
        #                  str(mlm.mapk(groundtruth_paintings_list_eval, predicted_paintings_list_eval, k)))

        print('**********************')
        # print('Params (m, r, d): ' + str(params['orb']['thr_matches']) + ',' + str(params['orb']['max_ratio']) +
        #              ', ' + str(params['orb']['max_distance']))
        print(f'MAP@{k}: {mlm.mapk(groundtruth_paintings_list_eval, predicted_paintings_list_eval, k)}')
