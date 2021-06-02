#!/bin/bash

python main.py --week4 data/BBDD data/qsd1_w4/ --remove_bg --remove_noise --remove_text --map_k 1,5 --verbose --max_paintings 3 --use_text --text_weight 0.1 --use_texture --texture_descriptor hog --texture_metric hellinger --texture_weight 0.6 --use_color --color_weight 0.3
