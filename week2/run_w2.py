import pickle
import os
import ml_metrics as mlm
import cv2 as cv

import week1.histogram as hist
import week1.masks as masks
import week1.evaluation as evaluation
import week1.bg_removal_methods as bg

def run():
    print('---------------------------------------------')

    # Path to bbdd and query datasets
    bbdd_path = 'data/BBDD'
    query_path = 'data/qsd2_w3/non_augmented'

    # Flags to select algorithms
    bg_removal = True
    text_detection = False

    # Test mode
    test = False

    # Parameters
    distance = "Hellinger"
    color_space = "RGB"
    k = 5 # Retrieve k most similar images
    n_bins = 8 # Number of bins per each histogram channel
    block_size = 16 # Block-based histogram
    method_compute_hist = "M1"
    method_bg = "M5" # Method to perform background removal

    # Path to results
    results_path = os.path.join(query_path, 'results_' + method_compute_hist)

    # If folder data/qsdX_wX/results doesn't exist -> create it
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if bg_removal:
        bg_results_path = os.path.join(results_path, 'bg_removal_' + method_bg)
        # If folder data/qsdX_wX/results/MX doesn't exist -> create it
        if not os.path.exists(bg_results_path):
            os.makedirs(bg_results_path)

    if text_detection:
        groundtruth_text_boxes_path = os.path.join(query_path, 'text_boxes.pkl')

    if not test:
        # Load groundtruth images of the query dataset
        groundtruth_paintings = pickle.load(open(os.path.join(query_path, "gt_corresps.pkl"), 'rb'))

    print('**********************')
    print('Dataset: {}, Method to compute histograms: {}, Background removal: {}, Text detection: {}'.format(query_path, method_compute_hist, bg_removal, text_detection))
    print('**********************')
    print("Computing bbdd histograms...", end=' ', flush=True)

    bbdd_histograms = hist.compute_bbdd_histograms(bbdd_path, method_compute_hist, n_bins, color_space, block_size)

    print("Done!")
    print('**********************')

    predicted_paintings_list = []
    groundtruth_paintings_list = []
    text_boxes = []

    # For each image of the query dataset, we remove the background (if needed), detect
    # the text bounding boxes (if needed), and compare the painting (or paintings, if there are two)
    # to each image of the bbdd dataset to retrieve the k most similar images
    for query_filename in sorted(os.listdir(query_path)):
        if query_filename.endswith('.jpg'):
            image_id = int(query_filename.replace('.jpg', ''))
            image_path = os.path.join(query_path, query_filename)

            # If needed, to store the painting/s (up to two) of an image
            paintings = []

            # If we need to remove the background
            if bg_removal:
                bg_mask_path = os.path.join(bg_results_path, query_filename.split(".")[0]+'.png')

                # Computes the background mask and returns the number of paintings in the image.
                # It also saves the background mask in the results path
                paintings_coords = masks.compute_bg_mask(image_path, bg_mask_path, method_bg, color_space)

                for painting_coords in paintings_coords:
                    # Gets the foreground image (painting). We pass the "idx" argument to define which painting we want.
                    # It also saves the foreground image in the results path
                    painting = masks.compute_fg(image_path, bg_mask_path, painting_coords, os.path.join(bg_results_path, query_filename))
                    paintings.append(painting)

            # If we don't need to remove the background --> painting = image
            else:
                paintings.append(cv.imread(image_path))

            # If needed, to store the text bounding boxes (up to two) of an image
            text_boxes_image = []

            # To store the paintings (up to two) k similar images
            predicted_paintings_per_image = []

            # For each painting
            for painting_id, painting in enumerate(paintings):

                # If we need to detect the text bounding box of the painting
                if text_detection:
                    [tlx, tly, brx, bry] = masks.detect_text_box(painting)

                    # If there are two paintings, when detecting the text bouning box of the
                    # second one we have to shift the coordinates so that they make sense in the initial image
                    if bg_removal:
                        tlx += paintings_coords[painting_id][0]
                        tly += paintings_coords[painting_id][1]
                        brx += paintings_coords[painting_id][0]
                        bry += paintings_coords[painting_id][1]

                    text_boxes_image.append([tlx, tly, brx, bry])

                    # Retrieves the k most similar images ignoring text bounding boxes
                    predicted_paintings = hist.get_k_images(painting, bbdd_histograms, text_boxes_image[painting_id],
                                                    method_compute_hist, k, n_bins, distance, color_space, block_size)

                else:
                    # Retrieves the k most similar images
                    predicted_paintings = hist.get_k_images(painting, bbdd_histograms, None,
                                                    method_compute_hist, k, n_bins, distance, color_space, block_size)

                predicted_paintings_per_image.append(predicted_paintings)

            if not test:
                print('Image: {}'.format(query_filename))
                for painting_id, groundtruth_painting in enumerate(groundtruth_paintings[image_id]):
                    print('-> Painting #{}'.format(painting_id))
                    print('    Groundtruth: {}'.format(groundtruth_painting))

                    # If we detected the painting
                    if len(predicted_paintings_per_image) > painting_id:
                        print('        {} most similar images: {}'.format(k, predicted_paintings_per_image[painting_id]))
                    else:
                        predicted_paintings_per_image.append([0, 0, 0, 0, 0])
                        print('        Painting not detected!!')

                print('----------------------')
                groundtruth_paintings_list.append(groundtruth_paintings[image_id])

            predicted_paintings_list.append(predicted_paintings_per_image)

            # Format of text_boxes: [[[tlx1, tly1, brx1, bry1], [tlx2, tly2, brx2, bry2]], [[tlx1, tly1, brx1, bry1]] ...]
            text_boxes.append(text_boxes_image)

    if not test:

        # Adapt the format of the "gt_corresps.pkl" to be able to evaluate the results...
        groundtruth_paintings_list_eval = []
        predicted_paintings_list_eval = []

        if 'qsd2_w2' in query_path or 'qsd2_w3' in query_path:
            for groundtruth_paintings_per_image in groundtruth_paintings_list:
                for groundtruth_painting in groundtruth_paintings_per_image:
                    groundtruth_paintings_list_eval.append([groundtruth_painting])
        else:
            groundtruth_paintings_list_eval = groundtruth_paintings_list

        for predicted_paintings_per_image in predicted_paintings_list:
            for predicted_paintings_per_painting in predicted_paintings_per_image:
                predicted_paintings_list_eval.append(predicted_paintings_per_painting)

        print("MAP@{}: {}".format(k, mlm.mapk(groundtruth_paintings_list_eval, predicted_paintings_list_eval, k)))
        print('**********************')

    predicted_paintings_path = os.path.join(results_path, 'result.pkl')
    predicted_paintings_outfile = open(predicted_paintings_path,'wb')
    pickle.dump(predicted_paintings_list, predicted_paintings_outfile)
    predicted_paintings_outfile.close()

    if text_detection:
        predicted_text_boxes_path = os.path.join(results_path, 'text_boxes.pkl')
        predicted_text_boxes_outfile = open(predicted_text_boxes_path,'wb')
        pickle.dump(text_boxes, predicted_text_boxes_outfile)
        predicted_text_boxes_outfile.close()

        # Text bounding boxes evaluation
        if not test:
            mean_iou = evaluation.mean_iou(query_path, groundtruth_text_boxes_path, predicted_text_boxes_path)
            print('**********************')
            print('Text bounding boxes evaluation: Mean IOU = {}'.format(mean_iou))
            print('**********************')

    # Background removal evaluation
    if bg_removal and not test:
        # The background masks and corresponding foregrounds are saved in 'data/qsdX_wX/results/MX'
        avg_precision, avg_recall, avg_f1 = masks.mask_average_evaluation(bg_results_path, query_path)
        print('**********************')
        print('Average --> Precision: {:.2f}, Recall: {:.2f}, F1-score: {:.2f}'.format(avg_precision, avg_recall, avg_f1))
        print('**********************')
    print('---------------------------------------------')
