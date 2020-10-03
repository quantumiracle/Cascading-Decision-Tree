#!/bin/bash

#--mem=20G
#--gres=gpu:0

hostname
echo $CUDA_VISIBLE_DEVICES

min=1
max=3
inter=1

## now loop through the above array
for ((m=2; m <= 4; m+=1));
do
    for ((i=min; i <= max; i+=inter));
    do 
        echo python3 rl_train_compare_sdt.py --train --env='CartPole-v1' --method="sdt" --id="$i" --depth="$m"
        python3 rl_train_compare_sdt.py --train --env='CartPole-v1' --method="sdt" --id="$i" --depth="$m"
    done
done
