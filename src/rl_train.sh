#!/bin/bash

#--mem=20G
#--gres=gpu:0

hostname
echo $CUDA_VISIBLE_DEVICES

min=1
max=3
inter=1
# declare an array to loop through
declare -a methods=("sdt" "cdt" "mlp")
declare -a envs=("CartPole-v1" "LunarLander-v2"  "MountainCar-v0")

## now loop through the above array
for env in "${envs[@]}";
do
    for method in "${methods[@]}";
    do
        for ((i=min; i <= max; i+=inter));
        do 
            echo python3 rl_train.py --train --env="$env" --method="$method" --id="$i"
            python3 rl_train.py --train --env="$env" --method="$method" --id="$i"
        done
    done
done
