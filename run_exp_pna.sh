export CUDA_VISIBLE_DEVICES=2
for i in 1 4 
do
    nohup python3 run_exp_local.py -c config/pna/multitask_pna_gnn$((i)).yaml &
    sleep 1
done

export CUDA_VISIBLE_DEVICES=1
for i in 3 2
do
    nohup python3 run_exp_local.py -c config/pna/multitask_pna_gnn$((i)).yaml &
    sleep 1
done

export CUDA_VISIBLE_DEVICES=3
for i in 5
do
    nohup python3 run_exp_local.py -c config/pna/multitask_pna_gnn$((i)).yaml &
    sleep 1
done

export CUDA_VISIBLE_DEVICES=0
for i in 24
do
    nohup python3 run_exp_local.py -c config/pna/multitask_pna_gnn$((i)).yaml &
    sleep 1
done
