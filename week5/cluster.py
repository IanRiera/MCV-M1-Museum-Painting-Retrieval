# import os
# import operator
# import cv2 as cv
# import numpy as np
# import pickle
# import ntpath
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# import os, glob, shutil
# import tensorflow as tf
# from collections import defaultdict
#
# from week4.histograms import  hog_histogram
# import week4.feature_descriptors as fd
#
# def get_color_images(images):
#     print("Using Colour for Clustering")
#     return np.array(np.float32(images).reshape(len(images), -1)/255), False
#
# def get_keras_prediction(images):
#     print("Using Keras for Clustering")
#     # Rudimentary Feature Extraction using Kerass
#     images = np.array(np.float32(images).reshape(len(images), -1)/255)
#
#     model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
#     predictions = model.predict(images.reshape(-1, 224, 224, 3))
#     pred_images = predictions.reshape(images.shape[0], -1)
#
#     return pred_images, False
#
# def hog_histogram_images(images):
#     print("Using HOG for Clustering")
#     hg_histograms = [hog_histogram(image, None) for image in images]
#
#     pred_image_hist = np.array(np.float32(hg_histograms).reshape(len(hg_histograms), -1)/255)
#
#     return pred_image_hist, False
#
# def orb_feature_descriptor_detection(images):
#     print("Using ORB for Clustering")
#     descriptors = np.array([])
#     for image in images:
#         _, orb_desc = fd.orb_descriptor(image, False)
#         if orb_desc is not None:
#             descriptors = np.append(descriptors, orb_desc)
#
#     x = np.float32(descriptors)
#
#     y = x.reshape(len(descriptors), -1)/255
#
#     descriptors = np.array(y)
#
#     return descriptors, False
#
# def cluster(bbdd_list, clusters=2,  technique='hog'):
#     # load the image and convert it from BGR to RGB so that
#     # we can dispaly it with matplotlib
#     images = [cv.resize(cv.imread(file), (224, 224)) for file in bbdd_list]
#     images = [cv.cvtColor(image, cv.COLOR_BGR2RGB) for image in images]
#
#     if technique=="colour":
#         predicted_images, do_itr = get_color_images(images) # Only Colour based matrices do not get very satisfactory results
#     elif technique=="keras":
#         predicted_images, do_itr = get_keras_prediction(images)
#     elif technique=="orb":
#         predicted_images, do_itr = orb_feature_descriptor_detection(images)
#     elif technique=="hog":
#         predicted_images, do_itr = hog_histogram_images(images)
#     else:
#         raise Exception("Invalid Clustering Technique")
#
#     clt = KMeans(n_clusters = clusters)
#
#     if do_itr:
#         for img_feature in predicted_images:
#             clt.fit(img_feature)
#     else:
#         print("Fitting Features")
#         clt.fit(predicted_images)
#
#     print("Predicting")
#     kpredictions = clt.predict(predicted_images)
#
#     print("Clustering with", technique)
#     for i in range(clusters):
#         os.makedirs("output/"+technique+"/cluster" + str(i))
#     for i in range(len(bbdd_list)):
#         shutil.copy2(bbdd_list[i], "output/"+technique+"/cluster"+str(kpredictions[i]))
#
#     # hist = centroid_histogram(clt)
#     # bar = plot_colors(hist, clt.cluster_centers_)
#
#
# # Some Utility Methods to visualize the clustering
# def centroid_histogram(clt):
# 	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
# 	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
# 	hist = hist.astype("float")
# 	hist /= hist.sum()
# 	return hist
#
# def plot_colors(hist, centroids):
# 	bar = np.zeros((50, 300, 3), dtype = "uint8")
# 	startX = 0
# 	for (percent, color) in zip(hist, centroids):
# 		endX = startX + (percent * 300)
# 		cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
# 			color.astype("uint8").tolist(), -1)
# 		startX = endX
#
# 	return bar
