#!/bin/bash
RHO_RES=1
THETA_RES=1
MIN_LINES=3
THR_INIT=($(seq 1000 -50 400))
THR_STEP=25
SHOW_H_LINES=$1
SHOW_V_LINES=$2

for l in ${MIN_LINES[@]}; do
	for t in ${THR_INIT[@]}; do
		if [ $# -eq 0 ]
			then
				python rotation_theta.py -r ${RHO_RES} -t ${THETA_RES} -m ${l} -i ${t} -s ${THR_STEP}
		else
			if [ "${1}" = "h" ]
				then
					python rotation_theta.py -r ${RHO_RES} -t ${THETA_RES} -m ${l} -i ${t} -s ${THR_STEP} --show_h_lines
			elif [ "${1}" = "v" ]
				then
					python rotation_theta.py -r ${RHO_RES} -t ${THETA_RES} -m ${l} -i ${t} -s ${THR_STEP} --show_v_lines
			elif [ "${1}" = "hv" ]
				then
					python rotation_theta.py -r ${RHO_RES} -t ${THETA_RES} -m ${l} -i ${t} -s ${THR_STEP} --show_h_lines --show_v_lines
			fi		
		fi
	done
done
