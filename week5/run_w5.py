import os
import sys
import argparse
import pickle

import week5.retrieval as retrieval
import week5.evaluation as evaluation
import week5.utils as utils
from week5 import cluster

def parse_args(args=sys.argv[2:]):
    parser = argparse.ArgumentParser(description='CBIR: Content Based Image Retrieval. MCV-M1-Project, Team 3')

    parser.add_argument('bbdd_path', type=str,
                        help='absolute/relative path to the bbdd dataset')

    parser.add_argument('query_path', type=str,
                        help='absolute/relative path to the query dataset')

    parser.add_argument('--test', action='store_true',
                        help='using a test dataset, so no groundtruth is provided')

    parser.add_argument('--map_k', type=lambda s: [int(item) for item in s.split(',')], default=[5],
                        help='retrieve K number/s of images')

    parser.add_argument('--remove_bg', action='store_true',
                        help='remove background from images in order to extract paintings')

    parser.add_argument('--max_paintings', type=int, default=3,
                        help='maximum number of paintings to extract from an image after removing the background')

    parser.add_argument('--remove_text', action='store_true',
                        help='remove text bounding boxes from images')

    parser.add_argument('--remove_noise', action='store_true',
                        help='remove noise from noisy images')

    parser.add_argument('--rotated', action='store_true',
                        help='rotated paintings (week5)')

    parser.add_argument('--use_color', action='store_true',
                        help='use color descriptor')

    parser.add_argument('--use_texture', action='store_true',
                        help='use texture descriptor/s')

    parser.add_argument('--use_text', action='store_true',
                        help='use text descriptor')

    parser.add_argument('--color_descriptor', type=str, default='rgb_3d_blocks',
                        choices=['rgb_3d_blocks', 'rgb_3d_multiresolution'],
                        help='color descriptor used')

    parser.add_argument('--texture_descriptor', type=lambda s: [item for item in s.split(',')], default='dct_blocks',
                        # choices=['dct_blocks', 'dct_multiresolution', 'lbp_blocks', 'lbp_multiresolution',
                        #          'hog', 'wavelet', 'hog_blocks', 'hog_multiresolution', 'wavelet_blocks', 'wavelet_multiresolution'],
                        help='texture descriptor used')

    parser.add_argument('--color_weight', type=float, default=0.33,
                        help='weight for the color descriptor')

    parser.add_argument('--texture_weight', type=lambda s: [float(item) for item in s.split(',')], default=[0.33],
                        help='weight for the texture descriptor')

    parser.add_argument('--text_weight', type=float, default=0.0,
                        help='weight for the text descriptor')

    parser.add_argument('--color_metric', type=str, default='hellinger',
                        choices=['hellinger', 'intersection', 'chi-squared', 'correlation'],
                        help='distance metric to compare images')

    parser.add_argument('--texture_metric', type=lambda s: [item for item in s.split(',')], default=['correlation'],
                        # choices=['hellinger', 'intersection', 'chi-squared', 'correlation'],
                        help='distance metric to compare images')

    parser.add_argument('--text_metric', type=str, default='Levensthein',
                        choices=['Levensthein', 'Hamming', 'Damerau'],
                        help='distance metric to compare images')

    parser.add_argument('--use_orb', action='store_true',
                        help='use ORB to predict images')

    parser.add_argument('--thr_matches', type=int, default=4,
                        help='minimum number of key point matches to match two paintings')

    parser.add_argument('--max_ratio', type=str, default='0.8',
                        help='max ratio used in match_descriptors function')

    parser.add_argument('--max_distance', type=str, default='1.2',
                        help='max distance used in match_descriptors function')

    parser.add_argument('--verbose', action='store_true',
                        help='increase output verbosity: show k similar images per each query image')

    parser.add_argument('--cluster_images', type=str,
                        help='Cluster Images')

    args = parser.parse_args(args)
    return args

def args_to_params(args):
    results_path = os.path.join(args.query_path, 'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    params = {
        'lists': None, 'paths': None, 'orb': None, 'color': None, 'texture': None, 'text': None, 'augmentation': None
    }
    params['paths'] = {
        'bbdd': args.bbdd_path,
        'query': args.query_path,
        'results': results_path
    }
    if args.use_color:
        params['color'] = {
            'descriptor': args.color_descriptor,
            'weight': args.color_weight,
            'metric': args.color_metric
        }
    if args.use_texture:
        params['texture'] = {
            'descriptor': args.texture_descriptor,
            'weight': args.texture_weight,
            'metric': args.texture_metric
        }
    if args.use_text:
        params['text'] = {
            'weight': args.text_weight,
            'metric': args.text_metric
        }
    if True in (args.remove_bg, args.remove_text, args.remove_noise, args.rotated):
        params['augmentation'] = {
            'bg': args.remove_bg,
            'max_paintings': args.max_paintings,
            'text': args.remove_text,
            'noise': args.remove_noise,
            'rotated': args.rotated
        }
    if args.use_orb:
        params['orb'] = {
            'thr_matches': args.thr_matches,
            'max_ratio': args.max_ratio,
            'max_distance': args.max_distance
        }
    if not True in (args.use_color, args.use_texture, args.use_text, args.use_orb, bool(args.cluster_images)):
        sys.exit('[ERROR] No descriptor method specified')

    return params

def lists_to_params(params, bbdd_list, query_list):
    params['lists'] = {
        'bbdd': bbdd_list,
        'query': query_list
    }
    return params

def run():
    args = parse_args()
    params = args_to_params(args)
    print(params)

    k = args.map_k

    bbdd_list = utils.path_to_list(params['paths']['bbdd'], extension='jpg')
    query_list = utils.path_to_list(params['paths']['query'], extension='jpg')

    params = lists_to_params(params, bbdd_list, query_list)

    if args.cluster_images:
        print("Clusters:=", max(k))
        for technique in args.cluster_images.split(","):
            cluster.cluster(bbdd_list, max(k), technique)
    else:
        paintings_predicted_list = retrieval.get_k_images(params, k=max(k))

        utils.save_pickle(os.path.join(params['paths']['results'], 'result.pkl'), paintings_predicted_list)

        if not args.test:
            evaluation.evaluate(params, k, verbose=args.verbose)

        else:
            evaluation.output_predicted_paintings_test(params, paintings_predicted_list, max(k))
