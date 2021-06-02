import pickle
import os
import ml_metrics as mlm

import week1.histogram as hist
import week1.masks as masks

def run():
    print('*********************************************')

    bbdd_path = '../data/BBDD'
    query_path = '../data/qsd1_w1'

    # load groundtruth images of the query dataset
    groundtruth_images = pickle.load(open(os.path.join(query_path, "gt_corresps.pkl"), 'rb'))

    # parameters: k most similar images, n_bins...
    k = 5
    n_bins = 8
    distance = "Hellinger"
    color_space = "RGB"

    print("Computing bbdd histograms...", end=' ', flush=True)
    bbdd_histograms = hist.compute_bbdd_histograms(bbdd_path, n_bins, color_space)
    print("Done!")
    print('----------------------')

    groundtruth_images_list = []
    predicted_images_list = []

    for query_filename in sorted(os.listdir(query_path)):
        if query_filename.endswith('.jpg'):
            image_id = int(query_filename.replace('.jpg', ''))
            predicted_images = hist.get_k_images(os.path.join(query_path, query_filename),
                                    bbdd_histograms, k, n_bins, distance, color_space)

            print('Image: {}, Groundtruth: {}'.format(query_filename, groundtruth_images[image_id]))
            print('{} most similar images: {}'.format(k, predicted_images))
            print('----------------------')

            groundtruth_images_list.append(groundtruth_images[image_id])
            predicted_images_list.append(predicted_images)

    print("MAP@{}: {}".format(k, mlm.mapk(groundtruth_images_list, predicted_images_list, k)))
    
    results_filename = query_path+'/result.pkl'
    outfile = open(results_filename,'wb')
    pickle.dump(predicted_images_list,outfile)
    outfile.close()

    # Background removal main:

    query_path_2 = '../data/qsd2_w1'
    method = "M0"
    # color_space = "RGB"

    masks.compute_masks(query_path_2, method, color_space)

    # the calculated masks and foregrounds will be save in 'data/qsd2_w1/results_Mx/'
    results_path = os.path.join(query_path_2+'/results_'+method+'/')

    avg_precision, avg_recall, avg_f1 = masks.mask_average_evaluation(results_path,query_path_2, method)

    print('-----------------------------------')
    print('Average --> Precision: {:.2f}, Recall: {:.2f}, F1-score: {:.2f}'.format(avg_precision, avg_recall, avg_f1))
    print('-----------------------------------')

    masks.compute_foregrounds(query_path_2,results_path, method)


    predicted_images_list_2 = []
    groundtruth_images_list_2 = []

    # load groundtruth images of the query dataset
    groundtruth_images_2 = pickle.load(open(os.path.join(query_path_2, "gt_corresps.pkl"), 'rb'))

    # find the k most similar for the foreground
    for query_filename in sorted(os.listdir(results_path)):
        if query_filename.endswith('.jpg'):
            image_id = int(query_filename.split(".")[0])
            predicted_images_2 = hist.get_k_images(os.path.join(results_path, query_filename),
                                    bbdd_histograms, k, n_bins, distance, color_space)
            print('Image: {}, Groundtruth: {}'.format(query_filename, groundtruth_images_2[image_id]))
            print('{} most similar images: {}'.format(k, predicted_images_2))
            print('----------------------')

            groundtruth_images_list_2.append(groundtruth_images_2[image_id])
            predicted_images_list_2.append(predicted_images_2)
            
    print("MAP@{}: {}".format(k, mlm.mapk(groundtruth_images_list_2, predicted_images_list_2, k))) 

    results_filename_2 = results_path+'result.pkl'
    outfile_2 = open(results_filename_2,'wb')
    pickle.dump(predicted_images_list_2,outfile_2)
    outfile_2.close()            
    
    print('*********************************************')
