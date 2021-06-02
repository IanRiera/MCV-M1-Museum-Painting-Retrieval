import pickle
import os
import ml_metrics as mlm
import cv2 as cv
import numpy as np
import operator

from tqdm import tqdm

import week3.imageToText as itt

import week3.histogram_w3 as hist
import week1.masks as masks
import week1.evaluation as evaluation
import week1.bg_removal_methods as bg
import week3.noise_removal as nr

TEXTURE_DESCRIPTORS_DISTANCES = {
    "LBP": "Correlation",
    "DCT": "Hellinger",
    "HOG": "",
    "WAVELET": ""
}
    ##------Oscar struct proposal----Future improvement
    # # Initialize parameters (color_ponderation, etc etc) and then create this struct:
    # params = {"color": None, "texture": None, "text": None, "bg_removal": None}
    #
    # if color_retrieval:
    #     params["color"] = {"weight": color_ponderation, "distance_metric": "Hellinger",
    #                        "color_space": "RGB", "n_bins": 8}
    # # We may want to combine different texture descriptors (e.g. LBP + DCT), so we need to add more than one ponderation/descriptor here
    # if texture_retrieval:
    #     params["texture"] = {"weight": texture_ponderation, "descriptor": texture_descriptor,
    #                          "distance_metric": TEXTURE_DESCRIPTORS_DISTANCES[texture_descriptor]}
    #
    # if text_retrieval:
    #     params["text"] = {"weight": text_ponderation, "distance_metric": "????"}
    #
    # if bg_removal:
    #     params["bg_removal"] = {"method": "M5"}
    #
    # # Then in compute_bbdd_histograms function (for example) we can only pass the params struct
    # # and check if a descriptor is needed (e.g. if params["texture"] is not None) and then access to the param
    # # values like params["texture"]["descriptor"]

def run():
    print('---------------------------------------------')


    # Path to bbdd and query datasets
    bbdd_path = 'data/BBDD'
    query_path = 'data/qsd1_w3'

    # Flags to select algorithms and ponderations
    bg_removal = False
    method_bg = "M5" # Method to perform background removal

    # Test mode
    test = False

    color_retrieval = False
    text_retrieval = False
    texture_retrieval = True

    text_ponderation = 1
    color_ponderation = 1
    texture_ponderation = 1

    # Color Parameters
    distance = "Hellinger"
    color_space = "RGB"
    k = 10 # Retrieve k most similar images
    n_bins = 8 # Number of bins per each histogram channel
    block_size = 16 # Block-based histogram
    method_compute_hist = "M1"
    method_texture = "M2"

    # Path to results
    results_path = os.path.join(query_path, 'results')

    # If folder data/qsdX_wX/results doesn't exist -> create it
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if bg_removal:
        bg_results_path = os.path.join(results_path, 'bg_removal_' + method_bg)
        # If folder data/qsdX_wX/results/MX doesn't exist -> create it
        if not os.path.exists(bg_results_path):
            os.makedirs(bg_results_path)

    if not test:
        # Load groundtruth images of the query dataset
        groundtruth_paintings = pickle.load(open(os.path.join(query_path, "gt_corresps.pkl"), 'rb'))
        groundtruth_text_boxes_path = os.path.join(query_path, 'text_boxes.pkl')

    if color_retrieval:

        print('**********************')
        print('Dataset: {}, Background removal: {}, Color retrieval: {}, Texture retrieval:{}, Text retrieval: {}'.format(query_path, bg_removal, color_retrieval,texture_retrieval,text_retrieval))
        print('**********************')
        print("Computing bbdd histograms...", end=' ', flush=True)

        bbdd_histograms = hist.compute_bbdd_histograms(bbdd_path, method_compute_hist, n_bins, color_space, block_size)

        print("Done!")
        print('**********************')

    if texture_retrieval:

        print('**********************')
        print('Dataset: {}, Background removal: {}, Color retrieval: {}, Texture retrieval:{}, Text retrieval: {}'.format(query_path, bg_removal, color_retrieval,texture_retrieval,text_retrieval))
        print('**********************')
        print("Computing bbdd textures...", end=' ', flush=True)

        bbdd_texture = hist.compute_bbdd_histograms(bbdd_path, method_texture, n_bins, color_space, block_size)

        print("Done!")
        print('**********************')

    if text_retrieval:

        print('**********************')
        print('Dataset: {}, Background removal: {}, Color retrieval: {}, Texture retrieval:{}, Text retrieval: {}'.format(query_path, bg_removal, color_retrieval,texture_retrieval,text_retrieval))
        print('**********************')
        print("Collecting bbdd texts...", end=' ', flush=True)

        bbdd_texts=itt.get_bbdd_texts(bbdd_path)

        print("Done!")
        print('**********************')


    predicted_paintings_list = []
    groundtruth_paintings_list = []
    text_boxes = []

    # For each image of the query dataset, we remove the background (if needed), denoise the image,
    # detect the text bounding boxes , and compare the painting (or paintings, if there are two)
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

            # To store the text bounding boxes (up to two) of an image
            text_boxes_image = []

            # To store the paintings (up to two) k similar images
            predicted_paintings_per_image = []

            # For each painting
            for painting_id, painting in enumerate(paintings):

                # First we denoise the painting
                painting=nr.denoiseImage(painting)
                # painting=nr.denoise_image_wavelet1(painting)
                # print(painting.shape)
                #To detect the text bounding box of the painting
                [tlx, tly, brx, bry],_ = masks.detect_text_box(painting)


                # We have to extract the text for each image and save it into a textfile
                # one painting per line
                painting_text = itt.get_text(painting,[tlx, tly, brx, bry])

                # # If there are two paintings, when detecting the text bouning box of the
                # # second one we have to shift the coordinates so that they make sense in the initial image
                # if bg_removal:
                #     tlx += paintings_coords[painting_id][0]
                #     tly += paintings_coords[painting_id][1]
                #     brx += paintings_coords[painting_id][0]
                #     bry += paintings_coords[painting_id][1]

                text_boxes_image.append([tlx, tly, brx, bry])
                predicted_text_path = os.path.join(results_path, query_filename.replace('.jpg', '.txt'))
                f= open(predicted_text_path,"a+")
                f.write(painting_text)
                f.close()

                if text_retrieval:
                # Retrieves the k most similar images ignoring text bounding boxes
                    predicted_text_paintings, author_list,text_distances = itt.get_k_images(painting, [tlx, tly, brx, bry],bbdd_texts,k=10,distance_metric="Levensthein")
                    predicted_paintings_per_image.append(predicted_text_paintings)


                if color_retrieval:
                # Retrieves the k most similar images ignoring text bounding boxes
                    predicted_color_paintings,color_distances = hist.get_k_images(painting, bbdd_histograms, [tlx, tly, brx, bry],
                                                method_compute_hist, k, n_bins, distance, color_space, block_size)


                if texture_retrieval:
                    predicted_texture_paintings,texture_distances = hist.get_k_images(painting, bbdd_texture, [tlx, tly, brx, bry],
                                                method_texture, k, n_bins, distance, color_space, block_size)
                    #do the stuff, get texture_distances


                lam1=color_ponderation*int(color_retrieval== True)
                lam2=texture_ponderation*int(texture_retrieval== True)
                lam3=text_ponderation*int(text_retrieval== True)

                weighted_distances={}
                if color_retrieval:
                    if text_retrieval:
                        if texture_retrieval:
                            for key in color_distances:
                                weighted_distances[key]=lam1*color_distances[key]+lam2*texture_distances[key]+lam3*text_distances[key]
                        else:
                            for key in color_distances:
                                weighted_distances[key]=lam1*color_distances[key]+lam3*text_distances[key]
                    elif texture_retrieval:
                        for key in color_distances:
                            weighted_distances[key]=lam1*color_distances[key]+lam2*texture_distances[key]
                    else:
                        weighted_distances=color_distances
                elif texture_retrieval:
                    if text_retrieval:
                        for key in texture_distances:
                            weighted_distances[key]=lam2*texture_distances[key]+lam3*text_distances[key]
                    else:
                        weighted_distances=texture_distances
                elif text_retrieval:
                    weighted_distances=text_distances
                else:
                    print("NO METHOD SELECTED!!!")


                k_predicted_images = (sorted(weighted_distances.items(), key=operator.itemgetter(1), reverse=False))[:k]

                predicted_paintings= [predicted_image[0] for predicted_image in k_predicted_images]

                predicted_paintings_per_image.append(predicted_paintings)


            predicted_paintings_list.append(predicted_paintings_per_image)

            # # Format of text_boxes: [[[tlx1, tly1, brx1, bry1], [tlx2, tly2, brx2, bry2]], [[tlx1, tly1, brx1, bry1]] ...]
            # text_boxes.append(text_boxes_image)

            if not test:
                print('Image: {}'.format(query_filename))
                for painting_id, groundtruth_painting in enumerate(groundtruth_paintings[image_id]):
                    print('-> Painting #{}'.format(painting_id))
                    print('    Groundtruth: {}'.format(groundtruth_painting))

                    # If we detected the painting
                    if len(predicted_paintings_per_image) > painting_id:
                        print('        {} most similar images: {}'.format(k, predicted_paintings_per_image[painting_id]))
                    else:
                        print('        Painting not detected!!')
                        predicted_paintings_per_image.append([0,0,0,0,0])

                print('----------------------')
                groundtruth_paintings_list.append(groundtruth_paintings[image_id])


    #-------EVALUATION AREA----------

    if not test:

        # Adapt the format of the "gt_corresps.pkl" to be able to evaluate the results...
        groundtruth_paintings_list_eval = []
        predicted_paintings_list_eval = []

        if 'qsd2_w3' in query_path:
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


    # predicted_text_boxes_path = os.path.join(results_path, 'text_boxes.pkl')
    # predicted_text_boxes_outfile = open(predicted_text_boxes_path,'wb')
    # pickle.dump(text_boxes, predicted_text_boxes_outfile)
    # predicted_text_boxes_outfile.close()
    #
    # # Text bounding boxes evaluation
    # if not test:
    #     mean_iou = evaluation.mean_iou(query_path, groundtruth_text_boxes_path, predicted_text_boxes_path)
    #     print('**********************')
    #     print('Text bounding boxes evaluation: Mean IOU = {}'.format(mean_iou))
    #     print('**********************')

    # Background removal evaluation
    if bg_removal and not test:
        # The background masks and corresponding foregrounds are saved in 'data/qsdX_wX/results/MX'
        avg_precision, avg_recall, avg_f1 = masks.mask_average_evaluation(bg_results_path, query_path)
        print('**********************')
        print('Average --> Precision: {:.2f}, Recall: {:.2f}, F1-score: {:.2f}'.format(avg_precision, avg_recall, avg_f1))
        print('**********************')
    print('---------------------------------------------')
