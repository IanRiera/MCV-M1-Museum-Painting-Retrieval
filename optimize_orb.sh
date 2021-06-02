#!/bin/bash
THR_MATCHES=($(seq 5 10))
MAX_RATIO=($(seq 0.4 0.1 1.4))
MAX_DISTANCE=($(seq 0.4 0.1 1.4))

for m in ${THR_MATCHES[@]}; do
	for r in ${MAX_RATIO[@]}; do
		for d in ${MAX_DISTANCE[@]}; do
			python main.py --week5 data/BBDD data/qsd1_w5/ --remove_bg --remove_noise --remove_text --max_paintings 3 --rotated --map_k 1 --use_orb --thr_matches ${m} --max_ratio ${r} --max_distance ${d} 
		done
	done
done
