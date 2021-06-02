#!/bin/bash

python main.py --week4 data/BBDD data/qsd1_w3/ --remove_noise --use_texture --texture_descriptor lbp_blocks --texture_metric chi-squared --map_k 1,5 --verbose
