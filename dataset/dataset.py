import pickle
import os
import os.path as osp
import glob
import networkx as nx
import numpy as np
import torch.utils.data
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import degree
from utils.arg_helper import get_config
import time
from tqdm import tqdm
import random
from utils.data_helper import DataListLoader, DataLoader

def multitask_data(config, split="train",
                   shuffle=False,
                   parallel=False,
                   meta_data_path=None,
                   random_init=True,
                   meta_copy=1,
                   num_module=1):
    tick = time.time()
    data_list = []

    split = split
    scale_data_path = config.dataset.test_path
    if 'train' in split or 'val' in split:
        batch_size = config.train.batch_size
        data_path = config.dataset.data_path
    elif 'test_test' in split:
        data_path = config.dataset.test_path
        batch_size = 1
    else:
        data_path = config.dataset.test_path
        batch_size = config.test.batch_size

    print(split, batch_size)
    data_files = sorted(glob.glob(os.path.join(data_path, split, '*.pt')))
    scale_data_files = sorted(glob.glob(os.path.join(config.dataset.scale_path, '*.pt')))
    name_list = []
    file_list = []
    max_degree = -1
    gt_pts_g_scale = []
    gt_pts_n_scale = []
    for idx, data_file in tqdm(enumerate(scale_data_files)):
        graph_data = torch.load(data_file)
        gt_pts_n_scale += [graph_data['y'][0].data.reshape(-1, 3).numpy()]
        gt_pts_g_scale += [graph_data['y'][1].data.reshape(-1, 3).numpy()]
    gt_pts_g_scale = np.concatenate(gt_pts_g_scale, axis=0)
    gt_pts_n_scale = np.concatenate(gt_pts_n_scale, axis=0)

    max_sssp = gt_pts_n_scale[:, 0].max()
    max_graph_laplacian = gt_pts_n_scale[:, 1].max()
    max_eccentricity = gt_pts_n_scale[:, 2].max()
    max_is_connected = gt_pts_g_scale[:, 0].max()
    max_diameter = gt_pts_g_scale[:, 1].max()
    max_spectral_radius = gt_pts_g_scale[:, 2].max()
    for idx, data_file in tqdm(enumerate(data_files)):
        graph_data = torch.load(data_file)
        num_nodes_I = graph_data['adj'].shape[0]
        dense_adj = graph_data['adj']
        G = nx.from_numpy_array(dense_adj)

        if meta_data_path is not None:
            # GRU assignment initialization
            if random_init:  # random setting
                random_list = [m for m in
                               range(num_module)]  # GRU index list between 0 and the number of GRU modules
                node_idx_inv = [random.choice(random_list) for _ in range(num_nodes_I)]
                node_idx = [[] for ___ in range(num_module)]
                for index, ii in enumerate(node_idx_inv):
                    node_idx[ii].append(index)

            else:  # biased setting
                node_idx_inv = []
                for deg in nx.degree(G):
                    if num_module == 1:
                        node_idx_inv.append(0)
                    else:
                        if num_module == 2:
                            if deg[1] < 50:
                                node_idx_inv.append(0)
                            else:
                                node_idx_inv.append(1)
                        elif num_module == 3:
                            if deg[1] < 33:
                                node_idx_inv.append(0)
                            elif 33 <= deg[1] < 66:
                                node_idx_inv.append(1)
                            else:
                                node_idx_inv.append(2)
                        elif num_module == 5:
                            if deg[1] < 20:
                                node_idx_inv.append(0)
                            elif 20 <= deg[1] < 40:
                                node_idx_inv.append(1)
                            elif 40 <= deg[1] < 60:
                                node_idx_inv.append(2)
                            elif 60 <= deg[1] < 80:
                                node_idx_inv.append(3)
                            elif 80 <= deg[1]:
                                node_idx_inv.append(4)
                node_idx = [[] for ___ in range(num_module)]
                for index, ii in enumerate(node_idx_inv):
                    node_idx[ii].append(index)
        else:
            if random_init:
                node_idx_inv = [random.choice([m for m in range(num_module)]) for ii in range(num_nodes_I)]
                node_idx = [[] for ___ in range(num_module)]
                for index, ii in enumerate(node_idx_inv):
                    node_idx[ii].append(index)
            else:
                node_idx_inv = []
                for deg in nx.degree(G):
                    if num_module == 1:
                        node_idx_inv.append(0)

                    else:
                        if num_module == 2:
                            if deg[1] < 50:
                                node_idx_inv.append(0)
                            else:
                                node_idx_inv.append(1)
                        elif num_module == 3:
                            if deg[1] < 33:
                                node_idx_inv.append(0)
                            elif 33 <= deg[1] < 66:
                                node_idx_inv.append(1)
                            else:
                                node_idx_inv.append(2)
                        elif num_module == 5:
                            if deg[1] < 20:
                                node_idx_inv.append(0)
                            elif 20 <= deg[1] < 40:
                                node_idx_inv.append(1)
                            elif 40 <= deg[1] < 60:
                                node_idx_inv.append(2)
                            elif 60 <= deg[1] < 80:
                                node_idx_inv.append(3)
                            elif 80 <= deg[1]:
                                node_idx_inv.append(4)
        if graph_data['name'] is not None:
            name = graph_data['name']
        else:
            name = idx
        graph_norm_factor = torch.tensor([max_is_connected, max_diameter, max_spectral_radius])
        node_norm_factor = torch.tensor([max_sssp, max_graph_laplacian, max_eccentricity])
        y_n = torch.div(graph_data['y'][0], node_norm_factor)
        y_g = torch.div(graph_data['y'][1], graph_norm_factor)
        data = Data(x=graph_data['x'], edge_index=graph_data['edge_index'], y_n=y_n,
                    y_g=y_g, node_idx=node_idx,
                    node_idx_inv=node_idx_inv, adj=graph_data['adj'])
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))
        for m_copy in range(meta_copy):
            data_list.append(data)
            name_list.append(name)
        
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for g in data_list:
        d = degree(g.edge_index[1], num_nodes=g.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    if parallel:
        loader = DataListLoader(data_list, batch_size=batch_size)
    else:
        loader = DataLoader(data_list, batch_size=batch_size)

    print("data loading time: ", time.time() - tick)
    return loader, name_list, deg, file_list


if __name__ == '__main__':
    # config = yaml.load(open("config/node_gnn10.yaml", 'r'), Loader=yaml.FullLoader)
    config = get_config("config/multitask_gnn0_meta.yaml", sample_id="{:03d}".format(0))
    tik = time.time()
    train_loader, _ = multitask_data(config, split='train')
    print(time.time() - tik)
    for i in train_loader:
        print(i)


