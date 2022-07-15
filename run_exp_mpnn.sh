# +
export CUDA_VISIBLE_DEVICES=2
for i in 1 5
do
    nohup python3 run_exp_local.py -c config/mpnn/multitask_gnn_meta$((i)).yaml &
    sleep 2
done

export CUDA_VISIBLE_DEVICES=1
for i in 2 3
do
    nohup python3 run_exp_local.py -c config/mpnn/multitask_gnn_meta$((i)).yaml &
    sleep 2
done

export CUDA_VISIBLE_DEVICES=3
for i in 24
do
    nohup python3 run_exp_local.py -c config/mpnn/multitask_gnn_meta$((i)).yaml &
    sleep 2
done

export CUDA_VISIBLE_DEVICES=3
for i in 4
do
    nohup python3 run_exp_local.py -c config/mpnn/multitask_gnn_meta$((i)).yaml &
    sleep 2
done
