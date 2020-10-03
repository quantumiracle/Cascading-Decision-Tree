#!/bin/bash

#--mem=20G
#--gres=gpu:0

hostname
echo $CUDA_VISIBLE_DEVICES

min=1
max=3
inter=1

## now loop through the above array
for ((m=1; m <= 3; m+=1));
do
    for ((n=2; n <= 2; n+=1));
    do
        for ((i=min; i <= max; i+=inter));
        do 
            echo python3 rl_train_compare_cdt.py --train --env='CartPole-v1' --method="cdt" --id="$i" --fl_depth="$m" --dm_depth="$n"
            python3 rl_train_compare_cdt.py --train --env='CartPole-v1' --method="cdt" --id="$i" --fl_depth="$m" --dm_depth="$n"
        done
    done
done
