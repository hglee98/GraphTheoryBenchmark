#!/bin/sh
export CUDA_VISIBLE_DEVICES=2
for i in 1 3 13
do
    nohup python3 run_exp_local.py -c config/module_1/multitask_gnn_meta$((i))_max.yaml &
    sleep 2
done

export CUDA_VISIBLE_DEVICES=0
for i in 2 4
do
    nohup python3 run_exp_local.py -c config/module_1/multitask_gnn_meta$((i))_max.yaml &
    sleep 2
done

export CUDA_VISIBLE_DEVICES=3
for i in 5
do
    nohup python3 run_exp_local.py -c config/module_1/multitask_gnn_meta$((i))_max.yaml &
    sleep 2
done

export CUDA_VISIBLE_DEVICES=1
for i in 24
do
    nohup python3 run_exp_local.py -c config/module_1/multitask_gnn_meta$((i))_max.yaml &
    sleep 2
done
