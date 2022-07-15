import argparse
import yaml
import os


def main():

    exp_list = [
    'GNN_exp/Multitask/multi_level/pna_meta/PNAGnn_meta_001_06-12-40_meta_group_1_hidden_16_num_prop_2_not_concat_b_lr_0.003_batch__128_2module_meta_copy_1_R',
    'GNN_exp/Multitask/multi_level/pna_meta/PNAGnn_meta_001_06-12-40_meta_group_2_hidden_16_num_prop_2_not_concat_b_lr_0.003_batch__128_2module_meta_copy_1_R',
    'GNN_exp/Multitask/multi_level/pna_meta/PNAGnn_meta_001_06-12-40_meta_group_3_hidden_16_num_prop_2_not_concat_b_lr_0.003_batch__128_2module_meta_copy_1_R',
    'GNN_exp/Multitask/multi_level/pna_meta/PNAGnn_meta_001_06-12-40_meta_group_4_hidden_16_num_prop_2_not_concat_b_lr_0.003_batch__128_2module_meta_copy_1_R',
    'GNN_exp/Multitask/multi_level/pna_meta/PNAGnn_meta_001_06-12-40_meta_group_5_hidden_16_num_prop_2_not_concat_b_lr_0.003_batch__128_2module_meta_copy_1_R',
    'GNN_exp/Multitask/multi_level/pna_meta/PNAGnn_meta_001_06-12-40_meta_group_2_4_hidden_16_num_prop_2_not_concat_b_lr_0.003_batch__128_2module_meta_copy_1_R',
    ]
    for idx, exp_dir in enumerate(exp_list):
        config_path = os.path.join(exp_dir, "config.yaml")
        cfg = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

        cfg['dataset']['test_path'] = 'data_temp/multitask'
        cfg['dataset']['split'] = "test"
        cfg['exp_dir'] = exp_dir

        cfg['test']['test_model'] = os.path.join(exp_dir, "model_snapshot_best.pth")
        

        with open('config/multi_mpnn_gnn_test_Multitask{}.yaml'.format(idx), 'w+') as ymlfile:
            yaml.dump(cfg, ymlfile, explicit_start=True)

        if idx % 4 == 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        elif idx % 4 == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        elif idx % 4 == 2:
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        os.system("nohup python3 run_exp_local.py -c config/multi_mpnn_gnn_test_Multitask{}.yaml -t &".format(idx))


if __name__ == '__main__':
    main()


