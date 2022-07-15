# -*- coding: utf-8 -*-
from __future__ import (division, print_function)

import copy
from collections import defaultdict
from tqdm import tqdm

import torch.utils.data
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

from model import *
from dataset import *
from utils.logger import get_logger
from utils.train_helper import data_to_gpu, snapshot, load_model, EarlyStopper
from utils.myparallel import DataParallel

logger = get_logger('exp_logger')
EPS = float(np.finfo(np.float32).eps)
__all__ = ['NeuralInferenceRunner', 'MultitaskRunner']


def propose_new_structure(node_idx, node_idx_inv, ch_node=True):
    if ch_node:
        change_node = (np.random.rand() > 0.5)
    else:
        change_node = False

    if change_node:
        idx = -1
        while idx == -1 or node_idx_inv[idx] >= len(node_idx):
            idx = np.random.randint(len(node_idx_inv))
        # Remove from old
        old_module = node_idx_inv[idx]
        pos_in_old = node_idx[old_module].index(idx)
        del node_idx[old_module][pos_in_old]
        # Add to new
        new_module = old_module
        while new_module == old_module:
            new_module = np.random.randint(len(node_idx))
        node_idx_inv[idx] = new_module
        node_idx[new_module].append(idx)
    return node_idx, node_idx_inv


def propose_new_structure_batch(node_idx, node_idx_inv, ch_node=True):

    new_node_idx = []
    new_node_idx_inv = []

    for node_idx_batch, node_idx_inv_batch in zip(node_idx, node_idx_inv):
        new_node_idx_batch, new_node_idx_inv_batch = propose_new_structure(node_idx_batch, node_idx_inv_batch,
                                                                           ch_node=ch_node)
        new_node_idx.append(new_node_idx_batch)
        new_node_idx_inv.append(new_node_idx_inv_batch)

    return new_node_idx, new_node_idx_inv


def node_idx_to_batch(node_idx, node_idx_inv, num_node, num_module=2):
    if num_module == 1:
        batched_node_idx = [[]]
        for i_batch, data in enumerate(node_idx):
            batched_node_idx[0] += [__ + i_batch * num_node for __ in data[0]]
    elif num_module == 2:
        batched_node_idx = [[], []]
        for i_batch, data in enumerate(node_idx):
            batched_node_idx[0] += [__ + i_batch * num_node for __ in data[0]]
            batched_node_idx[1] += [__ + i_batch * num_node for __ in data[1]]
    batched_node_idx_inv = []
    for _ in node_idx_inv:
        batched_node_idx_inv += _
    return batched_node_idx, batched_node_idx_inv


# +
class MultitaskRunner(object):
    def __init__(self, config):
        self.config = config
        self.dataset_conf = config.dataset
        self.model_conf = config.model
        self.train_conf = config.train
        self.test_conf = config.test
        self.use_gpu = config.use_gpu
        self.gpus = config.gpus
        self.writer = SummaryWriter(config.save_dir)
        self.shuffle = config.train.shuffle
        self.parallel = config.model.name == "TorchGNN_MsgGNN_parallel"
        self.master_node = config.model.master_node if config.model.master_node is not None else False
        self.SSL = config.model.SSL
        self.train_pretext = config.model.train_pretext
        self.initial_temp = self.train_conf.init_temp
        self.num_module = config.model.num_module
        self.node_only = self.model_conf.node_only
        self.graph_only = self.model_conf.graph_only
        self.min_temp = 0.00001
        self.temp_change = 1.1
        self.SA_running_acc_rate = 1e-9
        self.SA_running_factor = 1e-9
        self.initial_acc = 0
        self.temp_slope_opt_steps = self.train_conf.max_epoch
        self.temp_update = self.model_conf.temp

    @property
    def train(self):
        train_begin_time = time.time()
        # create data loader
        torch.cuda.empty_cache()
        train_loader,  _,  degree_histo, _ = eval(self.dataset_conf.loader_name)(self.config, split='train', shuffle=self.shuffle,
                                                                   parallel=self.parallel, meta_data_path=
                                                                   self.dataset_conf.data_path.split('/')[-1],
                                                                   random_init=self.train_conf.random_init,
                                                                   meta_copy=self.train_conf.meta_copy,
                                                                   num_module=self.model_conf.num_module)
        val_loader, _, _, _ = eval(self.dataset_conf.loader_name)(self.config, split='val', shuffle=self.shuffle,
                                                               parallel=self.parallel, meta_data_path=
                                                               self.dataset_conf.data_path.split('/')[-1],
                                                               random_init=self.train_conf.random_init,
                                                               meta_copy=self.train_conf.meta_copy,
                                                               num_module=self.model_conf.num_module)

        # create models
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.model_conf.name == 'PNAGnn_meta':
            model = eval(self.model_conf.name)(self.config, deg=degree_histo)
        else:
            model = eval(self.model_conf.name)(self.config)

        # fix parameter(pretext)
        if self.SSL and not self.train_pretext:
            print("Fixing parameter")
            for name, param in model.named_parameters():
                if "output_func" not in name:
                    param.requires_grad = False
                if "pretext_output_func" in name:
                    param.requires_grad = True

        if self.use_gpu:
            if self.parallel:
                print("Using GPU dataparallel")
                print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
                model = DataParallel(model)
            else:
                print("Using single GPU")
                model = nn.DataParallel(model, device_ids=self.gpus)
                # model = DataParallel(model)
        model.to(device)
        print(model)

        # create optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.train_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.train_conf.lr,
                momentum=self.train_conf.momentum,
                weight_decay=self.train_conf.wd)
        elif self.train_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.train_conf.lr,
                weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        early_stop = EarlyStopper([0.0], win_size=10, is_decrease=False)

        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.train_conf.lr_decay_steps)

        # reset gradient
        optimizer.zero_grad()

        # resume training
        if self.train_conf.is_resume:
            load_model(model, self.config.train.resume_model, optimizer=optimizer, train_pretext=self.train_pretext)

        # ========================= Training Loop =============================#
        best_train_loss = np.inf
        results = defaultdict(list)
        temp = self.initial_temp

        node_module_hist = []

        node_idx_list = []
        node_idx_inv_list = []
        for data_graph in tqdm(train_loader):
            node_idx_list.append(data_graph['node_idx'])
            node_idx_inv_list.append(data_graph['node_idx_inv'])

        snapshot(
            model.module if self.use_gpu else model,
            optimizer,
            self.config,
            0,
            tag="init")
#         torch.backends.cudnn.enabled = False
        model.train()
        for epoch in range(self.train_conf.max_epoch):

            if self.temp_update:
                acc_rate = np.exp(self.initial_acc - 5. * epoch / self.temp_slope_opt_steps)
                if self.SA_running_acc_rate / self.SA_running_factor < acc_rate:  # Reject가 많아지면 발생확률 up
                    temp *= self.temp_change
                else:
                    temp = temp / self.temp_change

            #################################
            ####       RECORD HIST        ###
            #################################
            node_module_hist.append(copy.deepcopy(node_idx_list))
            results['temp_hist'] += [temp]
            results['SA_hist'] += [[acc_rate, self.SA_running_acc_rate, self.SA_running_factor]]

            train_loss_epoch = []
            val_loss_epoch = []
            true_reject_count = 0
            true_accept_count = 0
            false_accept_count = 0

            for idx_main, [data1, data2] in tqdm(enumerate(zip(train_loader, val_loader))):
                structure_idx = idx_main
                loss = 0
                train_loss_batch = []
                val_loss_batch = []
                ########################
                # SEARCH TRAIN STRUCTURE
                ########################
                if "TorchGNN_MsgGNN" not in self.model_conf.name:
                    data1['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()
                    data2['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()
                ##############################
                # LOAD STRUCTURE FROM VAL DATA
                ##############################
                old_node_idx, old_node_idx_inv = copy.deepcopy(node_idx_list[structure_idx]), copy.deepcopy(
                    node_idx_inv_list[structure_idx])
                node_idx, node_idx_inv = node_idx_to_batch(node_idx_list[structure_idx],
                                                           node_idx_inv_list[structure_idx], self.dataset_conf.num_node,
                                                           self.model_conf.num_module)
                
                if self.num_module != 1:
                    new_node_idx, new_node_idx_inv = propose_new_structure_batch(node_idx_list[structure_idx],
                                                                                 node_idx_inv_list[structure_idx],
                                                                                 ch_node=self.train_conf.ch_node)
                    new_node_idx_, new_node_idx_inv_ = node_idx_to_batch(new_node_idx, new_node_idx_inv
                                                                         , self.dataset_conf.num_node,
                                                                         self.model_conf.num_module)
                model.train()
                _, old_loss_train, old_loss_train_batch = model(data1['x'], data1['edge_index'],
                                                                target_n=data1['y_n'], target_g=data1['y_g'],
                                                                node_idx=node_idx, node_idx_inv=node_idx_inv,
                                                                batch=data1['batch'])
                if self.num_module != 1:
                    _, new_loss_train, new_loss_train_batch = model(data1['x'], data1['edge_index'],
                                                                    target_n=data1['y_n'], target_g=data1['y_g'],
                                                                    node_idx=new_node_idx_, node_idx_inv=new_node_idx_inv_,
                                                                    batch=data1['batch'])

                model.eval()
                with torch.no_grad():
                    _, old_loss_val, old_loss_val_batch = model(data2['x'], data2['edge_index'],
                                                                target_n=data2['y_n'], target_g=data2['y_g'],
                                                                node_idx=node_idx,
                                                                node_idx_inv=node_idx_inv,
                                                                batch=data2['batch'])
                if self.num_module != 1:
                    with torch.no_grad():
                        _, new_loss_val, new_loss_val_batch = model(data2['x'], data2['edge_index'],
                                                                    target_n=data2['y_n'], target_g=data2['y_g'],
                                                                    node_idx=new_node_idx_,
                                                                    node_idx_inv=new_node_idx_inv_,
                                                                    batch=data2['batch'])
                    ##############################
                    # DECIDE STRUCTURE
                    ##############################
                    for idx, old_loss, new_loss in zip([_ for _ in range(len(old_loss_train_batch))], old_loss_train_batch,
                                                       new_loss_train_batch):
                        if self.temp_update:
                            upt_factor = min(0.01, self.SA_running_acc_rate / self.SA_running_factor)
                            # prob_accept = np.exp((old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()) / temp)

                            if old_loss.data.cpu().numpy() < new_loss.data.cpu().numpy():
                                results['loss_change'] += [old_loss.data.cpu().numpy() - new_loss.data.cpu().numpy()]
                                # results['prob_accept'] += [prob_accept]
                            accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy() #  or np.random.rand() < prob_accept
                        else:
                            accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()

                        if accept:  # ACCEPT
                            if old_loss.data.cpu().numpy() < new_loss.data.cpu().numpy():  # update running frac of worse accepts
                                self.SA_running_factor = ((1 - upt_factor) * self.SA_running_factor + upt_factor)
                                self.SA_running_acc_rate = (
                                            (1 - upt_factor) * self.SA_running_acc_rate + upt_factor)  # ACC/FACTOR UP
                                false_accept_count += 1
                            else:
                                true_accept_count += 1

                            node_idx_list[structure_idx][idx] = new_node_idx[idx]
                            node_idx_inv_list[structure_idx][idx] = new_node_idx_inv[idx]
                            train_loss_batch += [float(new_loss_train.data.cpu().numpy())]
                            val_loss_batch += [float(new_loss_val.data.cpu().numpy())]
                            loss += new_loss_train_batch[idx]

                        else:  # REJECT
                            if old_loss.data.cpu().numpy() < new_loss.data.cpu().numpy():  # update running frac of worse accepts
                                self.SA_running_factor = ((1 - upt_factor) * self.SA_running_factor + upt_factor)
                                self.SA_running_acc_rate = (1 - upt_factor) * self.SA_running_acc_rate  # ACC/FACTOR DOWN

                            node_idx_list[structure_idx][idx] = old_node_idx[idx]
                            node_idx_inv_list[structure_idx][idx] = old_node_idx_inv[idx]
                            true_reject_count += 1
                            train_loss_batch += [float(old_loss_train.data.cpu().numpy())]
                            val_loss_batch += [float(old_loss_val.data.cpu().numpy())]
                            loss += old_loss_train_batch[idx]
                else:
                    for idx, old_loss in zip([_ for _ in range(len(old_loss_train_batch))],
                                              old_loss_train_batch):
                        train_loss_batch += [float(old_loss_train.data.cpu().numpy())]
                        val_loss_batch += [float(old_loss_val.data.cpu().numpy())]
                        loss += old_loss_train_batch[idx]
                model.train()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                results['train_loss_batch'] += [np.stack(train_loss_batch).mean()]
                results['val_loss_batch'] += [np.stack(val_loss_batch).mean()]

                train_loss_epoch += [np.stack(train_loss_batch).mean()]
                val_loss_epoch += [np.stack(val_loss_batch).mean()]

#             lr_scheduler.step()

            results['true_accept'] += [true_accept_count]
            results['false_accept'] += [false_accept_count]
            results['true_reject'] += [true_reject_count]

            results['train_loss'] += [np.stack(train_loss_epoch).mean()]
            results['val_loss'] += [np.stack(val_loss_epoch).mean()]

            mean_loss = np.stack(train_loss_epoch).mean()
            mean_loss_val = np.stack(val_loss_epoch).mean()

            # save best model
            if mean_loss < best_train_loss:
                best_train_loss = mean_loss
                snapshot(
                    model.module if self.use_gpu else model,
                    optimizer,
                    self.config,
                    epoch + 1,
                    tag="best")

            snapshot(
                model.module if self.use_gpu else model,
                optimizer,
                self.config,
                epoch + 1,
                tag="final")

            logger.info("Train/Val Loss @ epoch {:04d}  = {}, {}".format(epoch + 1, mean_loss, mean_loss_val))
            logger.info("Current Best Train Loss = {}".format(best_train_loss))

        with open(os.path.join(self.config.save_dir, 'node_module_hist.p'), "wb") as f:
            pickle.dump(node_module_hist, f)
            del node_module_hist

        print(np.array(results['hidden_state']).shape)
        results['best_train_loss'] += [best_train_loss]
        train_end_time = time.time()
        results['total_time'] = train_end_time - train_begin_time
        pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
        self.writer.close()
        logger.info("Best Train Loss = {}".format(best_train_loss))
        torch.cuda.empty_cache()
        return best_train_loss

    def test(self):
        print(self.dataset_conf.loader_name)
        print(self.dataset_conf.split)
        torch.cuda.empty_cache()
        tik = time.time()
        test_loader, name_list, degree_histo, file_list = eval(self.dataset_conf.loader_name)(self.config, split='test',
                                                                                   shuffle=False,
                                                                                   random_init=self.test_conf.random_init,
                                                                                   num_module=self.test_conf.num_module)

        # create models
        if self.model_conf.name == 'PNAGnn_meta':
            model = eval(self.model_conf.name)(self.config, deg=degree_histo)
        else:
            model = eval(self.model_conf.name)(self.config)
        load_model(model, self.test_conf.test_model, train_pretext=self.train_pretext)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.use_gpu:
            if self.parallel:
                print("Using GPU dataparallel")
                print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
                model = DataParallel(model)
            else:
                print("Using single GPU")
                model = nn.DataParallel(model, device_ids=self.gpus)
                # model = DataParallel(model)
        model.to(device)
        print(model)

        model.eval()
        if self.node_only:
            pred_pts_n = []
            gt_pts_n = []
        elif self.graph_only:
            pred_pts_g = []
            gt_pts_g = []
        else:
            pred_pts_n = []
            gt_pts_n = []
            pred_pts_g = []
            gt_pts_g = []
            
        test_loss = []
        node_module_hist = []
        loss_list = []
        node_idx_list = []
        node_idx_inv_list = []
        for data in tqdm(test_loader):
            node_idx_list.append(data['node_idx'])
            node_idx_inv_list.append(data['node_idx_inv'])

        for step in tqdm(range(self.config.test.optim_step), desc="META TEST"):

            node_hist = copy.deepcopy(node_idx_inv_list)
            node_module_hist.append(node_hist)

            loss_ = []
            if step != self.config.test.optim_step - 1:
                for idx_main, data in tqdm(enumerate(test_loader)):
                    old_node_idx, old_node_idx_inv = copy.deepcopy(node_idx_list[idx_main]), copy.deepcopy(
                        node_idx_inv_list[idx_main])
                    node_idx, node_idx_inv = node_idx_to_batch(node_idx_list[idx_main], node_idx_inv_list[idx_main],
                                                               self.dataset_conf.num_node, self.model_conf.num_module)

                    if "TorchGNN_MsgGNN" not in self.model_conf.name:
                        data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

                    with torch.no_grad():

                        pred, loss, loss_batch = model(data['x'], data['edge_index'],
                                                       target_n=data['y_n'], target_g=data['y_g'],
                                                       node_idx=node_idx, node_idx_inv=node_idx_inv
                                                       , batch=data['batch'])
                        if self.num_module != 1:
                            new_node_idx, new_node_idx_inv = propose_new_structure_batch(node_idx_list[idx_main],
                                                                                         node_idx_inv_list[idx_main],
                                                                                         self.test_conf.ch_node)
                            new_node_idx_, new_node_idx_inv_ = node_idx_to_batch(new_node_idx, new_node_idx_inv,
                                                                                 self.dataset_conf.num_node, 
                                                                                 self.model_conf.num_module)

                            pred_new, loss_new, new_loss_batch = model(data['x'], data['edge_index'],
                                                                       target_n=data['y_n'], target_g=data['y_g']
                                                                       , node_idx=new_node_idx_,
                                                                       node_idx_inv=new_node_idx_inv_, batch=data['batch'])

                            for idx, old_loss, new_loss in zip([_ for _ in range(len(loss_batch))], loss_batch,
                                                               new_loss_batch):

                                accept = new_loss.data.cpu().numpy() <= old_loss.data.cpu().numpy()

                                if accept:
                                    node_idx_list[idx_main][idx] = new_node_idx[idx]
                                    node_idx_inv_list[idx_main][idx] = new_node_idx_inv[idx]
                                    loss_ += [float(new_loss.data.cpu().numpy())]
                                else:
                                    node_idx_list[idx_main][idx] = old_node_idx[idx]
                                    node_idx_inv_list[idx_main][idx] = old_node_idx_inv[idx]
                                    loss_ += [float(old_loss.data.cpu().numpy())]
                        else:
                            for idx, old_loss in zip([_ for _ in range(len(loss_batch))], loss_batch):
                                loss_ += [float(old_loss.data.cpu().numpy())]
            else:
                logger.info("=======================================")
                logger.info("TEST")
                logger.info("=======================================")
                for idx_main, data in tqdm(enumerate(test_loader)):
                    old_node_idx, old_node_idx_inv = copy.deepcopy(node_idx_list[idx_main]), copy.deepcopy(
                        node_idx_inv_list[idx_main])
                    node_idx, node_idx_inv = node_idx_to_batch(node_idx_list[idx_main], node_idx_inv_list[idx_main],
                                                               self.dataset_conf.num_node, 
                                                                                 self.model_conf.num_module)

                    if "TorchGNN_MsgGNN" not in self.model_conf.name:
                        data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

                    with torch.no_grad():

                        pred, loss, loss_batch = model(data['x'], data['edge_index'],
                                                       target_n=data['y_n'], target_g=data['y_g'],
                                                       node_idx=node_idx, node_idx_inv=node_idx_inv
                                                       , batch=data['batch'])
                        if self.node_only:
                            pred_pts_n += [pred.data.cpu().numpy()]
                            gt_pts_n += [data['y_n'].data.cpu().reshape(-1, 3).numpy()]
                        elif self.graph_only:
                            pred_pts_g += [pred.data.cpu().reshape(-1, 3).numpy()]
                            gt_pts_g += [data['y_g'].data.cpu().reshape(-1, 3).numpy()]
                        else:
                            pred_pts_n += [pred[0].data.cpu().numpy()]
                            gt_pts_n += [data['y_n'].data.cpu().reshape(-1, 3).numpy()]
                            pred_pts_g += [pred[1].data.cpu().reshape(-1, 3).numpy()]
                            gt_pts_g += [data['y_g'].data.cpu().reshape(-1, 3).numpy()]
                        for idx, old_loss in zip([_ for _ in range(len(loss_batch))], loss_batch):
                            loss_ += [float(old_loss.data.cpu().numpy())]
            mean_loss = np.stack(loss_).mean()
            loss_list.append(mean_loss)
            logger.info("Test Loss @ epoch {:04d} = {}".format(step + 1, mean_loss))
        if self.node_only:
            pred_pts_n = np.concatenate(pred_pts_n, axis=0)
            gt_pts_n = np.concatenate(gt_pts_n, axis=0)
        elif self.graph_only:
            pred_pts_g = np.concatenate(pred_pts_g, axis=0)
            gt_pts_g = np.concatenate(gt_pts_g, axis=0)
        else:
            pred_pts_n = np.concatenate(pred_pts_n, axis=0)
            gt_pts_n = np.concatenate(gt_pts_n, axis=0)
            pred_pts_g = np.concatenate(pred_pts_g, axis=0)
            gt_pts_g = np.concatenate(gt_pts_g, axis=0)
            
        name_list = np.array(name_list)

        if self.node_only:
            np.savetxt(self.config.save_dir + '/pred_pts_n' + self.dataset_conf.split + '.csv', pred_pts_n, delimiter='\t')
            np.savetxt(self.config.save_dir + '/gt_pts_n' + self.dataset_conf.split + '.csv', gt_pts_n, delimiter='\t')
        elif self.graph_only:
            np.savetxt(self.config.save_dir + '/pred_pts_g' + self.dataset_conf.split + '.csv', pred_pts_g, delimiter='\t')
            np.savetxt(self.config.save_dir + '/gt_pts_g' + self.dataset_conf.split + '.csv', gt_pts_g, delimiter='\t')
        else:
            np.savetxt(self.config.save_dir + '/pred_pts_n' + self.dataset_conf.split + '.csv', pred_pts_n, delimiter='\t')
            np.savetxt(self.config.save_dir + '/gt_pts_n' + self.dataset_conf.split + '.csv', gt_pts_n, delimiter='\t')
            np.savetxt(self.config.save_dir + '/pred_pts_g' + self.dataset_conf.split + '.csv', pred_pts_g, delimiter='\t')
            np.savetxt(self.config.save_dir + '/gt_pts_g' + self.dataset_conf.split + '.csv', gt_pts_g, delimiter='\t')

        total_time = time.time() - tik

        with open(os.path.join(self.config.save_dir, "{}.txt".format(total_time)), 'wb') as f:
            pickle.dump(total_time, f)

        file_name = os.path.join(self.config.save_dir, "name.p")
        with open(file_name, 'wb') as f:
            pickle.dump(name_list, f)

        file_name = os.path.join(self.config.save_dir, "file.p")
        with open(file_name, 'wb') as f:
            pickle.dump(file_list, f)

        file_name = os.path.join(self.config.save_dir, "node_module_hist.p")
        with open(file_name, 'wb') as f:
            pickle.dump(node_module_hist, f)

        file_name = os.path.join(self.config.save_dir, "test_loss_list.p")
        with open(file_name, 'wb') as f:
            pickle.dump(loss_list, f)

        return loss_list


# -

class NeuralInferenceRunner(object):
    def __init__(self, config):
        self.config = config
        self.dataset_conf = config.dataset
        self.model_conf = config.model
        self.train_conf = config.train
        self.test_conf = config.test
        self.node_only = config.model.node_only
        self.graph_only = config.model.graph_only
        self.use_gpu = config.use_gpu
        self.gpus = config.gpus
        self.writer = SummaryWriter(config.save_dir)
        self.shuffle = config.train.shuffle
        self.parallel = config.model.name == "TorchGNN_MsgGNN_parallel"
        self.master_node = config.model.master_node if config.model.master_node is not None else False
        self.SSL = config.model.SSL
        self.train_pretext = config.model.train_pretext

    @property
    def train(self):
        train_begin_time = time.time()
        # create data loader
        torch.cuda.empty_cache()
        train_loader, _, degree_histo, _ = eval(self.dataset_conf.loader_name)(self.config, split='train'
                                                                               , shuffle=self.shuffle,
                                                                                parallel=self.parallel)
        val_loader, _, _, _ = eval(self.dataset_conf.loader_name)(self.config, split='val', shuffle=self.shuffle,
                                                                parallel=self.parallel)

        # create models
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.model_conf.name == 'PNAGnn':
            model = eval(self.model_conf.name)(self.config, deg=degree_histo)
        else:
            model = eval(self.model_conf.name)(self.config)
        if self.use_gpu:
            if self.parallel:
                print("Using GPU dataparallel")
                print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
                model = DataParallel(model)
            else:
                print("Using single GPU")
                model = nn.DataParallel(model, device_ids=self.gpus)
                # model = DataParallel(model)
        model.to(device)
        print(model)

        # create optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.train_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.train_conf.lr,
                momentum=self.train_conf.momentum,
                weight_decay=self.train_conf.wd)
        elif self.train_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.train_conf.lr,
                weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        early_stop = EarlyStopper([0.0], win_size=10, is_decrease=False)

        #reset gradient
        optimizer.zero_grad()

        # resume training
        if self.train_conf.is_resume:
            load_model(model, self.config.train.resume_model, optimizer=optimizer, train_pretext=self.train_pretext)

        # ========================= Training Loop =============================#
        iter_count = 0
        best_val_loss = np.inf
        results = defaultdict(list)
        for epoch in range(self.train_conf.max_epoch):
            # ===================== validation ============================ #
            model.eval()
            val_loss = []
            for data in tqdm(val_loader):
                with torch.no_grad():
                    _, loss = model(data['x'], data['edge_index'],
                                    target_n=data['y_n'], target_g=data['y_g']
                                    , batch=data['batch'])

                    val_loss += [float(loss.data.cpu().numpy())]

            val_loss = np.stack(val_loss).mean()
            results['val_loss'] += [val_loss]
            logger.info("Avg. Validation Loss = {} +- {}, {} epoch".format(val_loss, 0, epoch))
            self.writer.add_scalar('val_loss', val_loss, iter_count)

            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                snapshot(
                    model.module if self.use_gpu else model,
                    optimizer,
                    self.config,
                    epoch + 1,
                    tag="best")

            logger.info("Current Best Validation Loss = {}".format(best_val_loss))
                
            # ====================== training ============================= #
            model.train()
            train_loss = []
            for data in train_loader:
                # 0. clears all gradients.
                optimizer.zero_grad()
                # if self.use_gpu:
                if "TorchGNN_MsgGNN" not in self.model_conf.name:
                    data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

                # 1. forward pass
                # 2. compute loss
                if self.parallel:
                    _ = model(data)
                    target = torch.cat([i.y for i in data]).to(_.device)
                    loss = nn.KLDivLoss(reduction='batchmean')(_, target)
                else:
                    if self.SSL:
                        if not self.train_pretext:
                            _, loss = model(data['x'], data['edge_index'],
                                            target_n=data['y_n'], target_g=data['y_g']
                                            , batch=data['batch'])
                        else:
                            _, loss = model(data['x'], data['edge_index'],
                                            target_n=data['y_n'], target_g=data['y_g']
                                            , batch=data['batch'])
                    elif self.model_conf.name == "TorchGNN_multitask":
                        _, loss = model(data['x'], data['edge_index'],
                                        target_n=data['y_n'], target_g=data['y_g']
                                        , batch=data['batch'])
                    else:
                        _, loss = model(data['x'], data['edge_index'],
                                        target_n=data['y_n'], target_g=data['y_g']
                                        , batch=data['batch'])

                # 3. backward pass (accumulates gradients).
                loss.backward()
                # 4. performs a single update step.
                optimizer.step()

                train_loss += [float(loss.data.cpu().numpy())]
                iter_count += 1
            train_loss = np.stack(train_loss).mean()
            self.writer.add_scalar('train_loss', train_loss)
            results['train_loss'] += [train_loss]
            logger.info("Avg. train Loss = {} +- {}, {} epoch".format(train_loss, 0, epoch))
            # snapshot model
            if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
                logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
                snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1)

                
        print(np.array(results['hidden_state']).shape)
        results['best_val_loss'] += [best_val_loss]
        train_end_time = time.time()
        results['total_time'] = train_end_time - train_begin_time
        pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
        self.writer.close()
        logger.info("Best Validation Loss = {}".format(best_val_loss))

        return best_val_loss

    def test(self):
        print(self.dataset_conf.loader_name)
        print(self.dataset_conf.split)
        torch.cuda.empty_cache()
        # create data loader

        test_loader, name_list, degree_histo, file_list = eval(self.dataset_conf.loader_name)(self.config, split='test',
                                                                                shuffle=False)

        # create models
        if self.model_conf.name == 'PNAGnn':
            model = eval(self.model_conf.name)(self.config, test=True, deg=degree_histo)
        else:
            model = eval(self.model_conf.name)(self.config, test=True)

            
        if 'Gnn' in self.model_conf.name:
            load_model(model, self.test_conf.test_model, train_pretext=self.train_pretext)

        # create models
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.use_gpu:
            if self.parallel:
                print("Using GPU dataparallel")
                print('Let\'s use', torch.cuda.device_count(), 'GPUs!')
                model = DataParallel(model)
            else:
                print("Using single GPU")
                model = nn.DataParallel(model, device_ids=self.gpus)
        model.to(device)
        print(model)

        model.eval()
        test_loss = []
        pred_pts_n = []
        pred_pts_g = []
        gt_pts_n = []
        gt_pts_g = []
        state_hist = []
        for data in tqdm(test_loader):
            if "TorchGNN_MsgGNN" not in self.model_conf.name:
                data['idx_msg_edge'] = torch.tensor([0, 0]).contiguous().long()

            with torch.no_grad():
                prob, loss = model(data['x'], data['edge_index'],
                                   target_n=data['y_n'], target_g=data['y_g']
                                   , batch=data['batch'])

                loss = 0 if loss < 0 else float(loss.data.cpu().numpy())
                test_loss += [loss]

                if self.node_only:
                    pred_pts_n += [prob.data.cpu().numpy()]
                    gt_pts_n += [data['y_n'].data.cpu().reshape(-1, 3).numpy()]
                elif self.graph_only:
                    pred_pts_g +=[prob.data.cpu().reshape(-1, 3).numpy()]
                    gt_pts_g += [data['y_g'].data.cpu().reshape(-1, 3).numpy()]
                else:
                    pred_pts_n += [prob[0].data.cpu().numpy()]
                    gt_pts_n += [data['y_n'].data.cpu().reshape(-1, 3).numpy()]
                    pred_pts_g +=[prob[1].data.cpu().reshape(-1, 3).numpy()]
                    gt_pts_g += [data['y_g'].data.cpu().reshape(-1, 3).numpy()]

        test_loss = np.stack(test_loss).mean()
        logger.info("Avg. Test Loss = {} +- {}".format(test_loss, 0))

        if self.node_only:
            pred_pts_n = np.concatenate(pred_pts_n, axis=0)
            gt_pts_n = np.concatenate(gt_pts_n, axis=0)
        elif self.graph_only:
            pred_pts_g = np.concatenate(pred_pts_g, axis=0)
            gt_pts_g = np.concatenate(gt_pts_g, axis=0)
        else:
            pred_pts_n = np.concatenate(pred_pts_n, axis=0)
            gt_pts_n = np.concatenate(gt_pts_n, axis=0)
            pred_pts_g = np.concatenate(pred_pts_g, axis=0)
            gt_pts_g = np.concatenate(gt_pts_g, axis=0)
        name_list = np.array(name_list)
        state_hist = np.array(state_hist)

        if self.node_only:
            np.savetxt(self.config.save_dir + '/pred_pts_n' + self.dataset_conf.split + '.csv', pred_pts_n, delimiter='\t')
            np.savetxt(self.config.save_dir + '/gt_pts_n' + self.dataset_conf.split + '.csv', gt_pts_n, delimiter='\t')
        elif self.graph_only:
            np.savetxt(self.config.save_dir + '/pred_pts_g' + self.dataset_conf.split + '.csv', pred_pts_g, delimiter='\t')
            np.savetxt(self.config.save_dir + '/gt_pts_g' + self.dataset_conf.split + '.csv', gt_pts_g, delimiter='\t')
        else:
            np.savetxt(self.config.save_dir + '/pred_pts_n' + self.dataset_conf.split + '.csv', pred_pts_n, delimiter='\t')
            np.savetxt(self.config.save_dir + '/gt_pts_n' + self.dataset_conf.split + '.csv', gt_pts_n, delimiter='\t')
            np.savetxt(self.config.save_dir + '/pred_pts_g' + self.dataset_conf.split + '.csv', pred_pts_g, delimiter='\t')
            np.savetxt(self.config.save_dir + '/gt_pts_g' + self.dataset_conf.split + '.csv', gt_pts_g, delimiter='\t')

        file_name = os.path.join(self.config.save_dir, "name.p")

        with open(file_name, 'wb') as f:
            pickle.dump(name_list, f)

        file_name = os.path.join(self.config.save_dir, "file.p")
        with open(file_name, 'wb') as f:
            pickle.dump(file_list, f)

        file_name = os.path.join(self.config.save_dir, "state_hist.p")
        with open(file_name, 'wb') as f:
            pickle.dump(state_hist, f)

        return test_loss


