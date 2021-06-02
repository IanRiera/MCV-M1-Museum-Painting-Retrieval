import os
import cv2 as cv
import numpy as np
from tqdm import tqdm
import os

import multiprocessing.dummy as mp
from functools import partial
from itertools import repeat

import week4.masks as masks
import week4.histograms as histograms
import week4.text_boxes as text_boxes_detection
import week4.noise_removal as noise
import week4.utils as utils
import week4.feature_descriptors as feature_descriptors
import week4.sift as sift
import week4.image_to_text as text_detection

def image_to_paintings(image_path, params):
    img = cv.imread(image_path)
    image_id = utils.get_image_id(image_path)

    paintings=[img]
    text_boxes=[None]
    text_boxes_shift=[None]

    paintings_coords = [[0,0,0,0]]

    if params['remove'] is not None:
        if params['remove']['bg']:
            [paintings, paintings_coords] = masks.remove_bg(img, params, image_id)

        if params['remove']['noise']:
            paintings = noise.denoise_paintings(paintings, params, image_id)

        if params['remove']['text']:
            [paintings, text_boxes, text_boxes_shift] = text_boxes_detection.remove_text(paintings, paintings_coords, params, image_id)
            # for idx,painting in enumerate(paintings):
            #     if text_boxes[idx] is not None:
            #         text_detected=text_detection.get_text(painting,text_boxes[idx])
            #
            #         predicted_text_path = os.path.join(params['paths']['results'], '{}.txt'.format(image_id))
            #         with open(predicted_text_path,"a+") as f:
            #             f.write(text_detected+"\n")

    return [paintings, text_boxes, text_boxes_shift]


def get_k_images(params, k):

    pool_processes = 4

    paintings_predicted_list = []

    with mp.Pool(processes=pool_processes) as p:

        image_to_paintings_partial = partial(image_to_paintings, params=params)

        print('---Extracting paintings from images (optional: removing background or text)---')
        [paintings, text_boxes, text_boxes_shift] = zip(*list(tqdm(p.imap(image_to_paintings_partial,
                                                  [path for path in params['lists']['query']]),
                                                  total=len(params['lists']['query']))))


        utils.save_pickle(os.path.join(params['paths']['results'], 'text_boxes.pkl'), text_boxes_shift)


        all_distances = []

        if params['color'] is not None:
            compute_bbdd_histograms_partial = partial(histograms.compute_bbdd_histograms,
                                                      descriptor=params['color']['descriptor'])

            print('---Computing color bbdd_histograms---')
            bbdd_histograms = list(tqdm(p.imap(compute_bbdd_histograms_partial,
                                              [path for path in params['lists']['bbdd']]),
                                              total=len(params['lists']['bbdd'])))

            print('---Computing color query_histograms and distances---')
            color_distances = histograms.compute_distances(paintings, text_boxes, bbdd_histograms,
                                                           descriptor=params['color']['descriptor'],
                                                           metric=params['color']['metric'],
                                                           weight=params['color']['weight'])
            all_distances.append(color_distances)

        if params['texture'] is not None:
            for texture_id, texture_descriptor in enumerate(params['texture']['descriptor']):
                compute_bbdd_histograms_partial = partial(histograms.compute_bbdd_histograms,
                                                          descriptor=texture_descriptor)

                print('...Computing texture bbdd_histograms...')
                bbdd_histograms = list(tqdm(p.imap(compute_bbdd_histograms_partial,
                                                  [path for path in params['lists']['bbdd']]),
                                                  total=len(params['lists']['bbdd'])))

                print('---Computing texture query_histograms and distances---')
                texture_distances = histograms.compute_distances(paintings, text_boxes, bbdd_histograms,
                                                               descriptor=texture_descriptor,
                                                               metric=params['texture']['metric'][texture_id],
                                                               weight=params['texture']['weight'][texture_id])

                all_distances.append(texture_distances)

        if params['features'] is not None:

            if params['features']['orb']:

                print('---Computing ORB bbdd_histograms---')
                bbdd_descriptors = list(tqdm(p.imap(feature_descriptors.compute_bbdd_orb_descriptors,
                                                    [path for path in params['lists']['bbdd']]),
                                             total=len(params['lists']['bbdd'])))

                predicted_paintings_all = []
                print('---Computing ORB query_histograms and distances---')
                for image_id, paintings_image in tqdm(enumerate(paintings), total=len(paintings)):
                    predicted_paintings_image = []
                    for painting_id, painting in enumerate(paintings_image):
                        painting_kp, painting_des = feature_descriptors.orb_descriptor(painting)
                        if len(painting_kp) > 0:

                            match_descriptors_partial = partial(feature_descriptors.match_descriptors, query_des=painting_des)
                            matches = p.map(match_descriptors_partial, [kp_des for kp_des in bbdd_descriptors])

                            predicted_paintings = feature_descriptors.get_top_matches(matches)

                            if predicted_paintings is not None:
                                predicted_paintings_image.append(predicted_paintings[:k])
                            else:
                                predicted_paintings_image.append([-1])

                        else:
                            print('???????????????????????????????????????????????????????????????????????')
                            print(f'Image ID: {image_id}, Painting ID: {painting_id}')
                            predicted_paintings_image.append([-1])

                    predicted_paintings_all.append(predicted_paintings_image)

                return predicted_paintings_all

            if params['features']['sift']:
                print('NOT IMPLEMENTED')
                # paintings_predicted_list = sift()
                # match_dict = sift.process_query(query_list, bbdd_list)

            if params['features']['surf']:
                print('NOT IMPLEMENTED')
                # paintings_predicted_list = feature_descriptors.surf()

        if params['text'] is not None:
            print('...Computing text histograms and distances...')
            bbdd_texts = text_detection.get_bbdd_texts(params['paths']['bbdd'])

            text_distances = text_detection.compute_distances(paintings, text_boxes, bbdd_texts,
                                                            metric=params['text']['metric'],
                                                            weight=params['text']['weight'])


            all_distances.append(text_distances)

        for q in range(len(paintings)):
            qlist = []
            for sq in range(len(paintings[q])):
                dist = np.array(all_distances[0][q][sq])
                for f in range(1, len(all_distances)):
                    dist += all_distances[f][q][sq]
                nearest_indices = np.argsort(dist)[:k]
                result_list = [index for index in nearest_indices]
                qlist.append(result_list)
            paintings_predicted_list.append(qlist)

    return paintings_predicted_list
