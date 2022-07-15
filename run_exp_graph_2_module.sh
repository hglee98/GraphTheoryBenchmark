#!/bin/sh
# +
### Graph Level Experiments

export CUDA_VISIBLE_DEVICES=1

for i in 1 3
do
    nohup python3 run_exp_local.py -c config/module_2/graph_gnn$((i))_meta.yaml &
    sleep 2
done

export CUDA_VISIBLE_DEVICES=2

for i in 2 4 
do
    nohup python3 run_exp_local.py -c config/module_2/graph_gnn$((i))_meta.yaml &
    sleep 2
done

export CUDA_VISIBLE_DEVICES=4

for i in 24
do
    nohup python3 run_exp_local.py -c config/module_2/graph_gnn$((i))_meta.yaml &
    sleep 2
done

export CUDA_VISIBLE_DEVICES=7

for i in 5
do
    nohup python3 run_exp_local.py -c config/module_2/graph_gnn$((i))_meta.yaml &
    sleep 2
done
