from tqdm import tqdm
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
from collections import Counter, defaultdict
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import math


def degree(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degree_count = Counter(degree_sequence)
    deg, cnt = zip(*degree_count.items())
    return degree_sequence, deg, cnt


def func(x, a, b, c):
    return a * (x-b)**2 + c


def save_fig(f, name):
    f.savefig(name, bbox_inches='tight')


def color(test_path, minmax1, minmax2, num_share, number_of_model_plot, f):
    cmap = plt.cm.bwr
    for x, _ in tqdm(enumerate(test_path), desc="outer"):
        train_path = '/'.join(_.split('/')[:-1])

        XYL = pickle.load(open(os.path.join(_, 'XYL.p'), "rb"))
        num_node = 100
        color = XYL[:, 2]
        loss = np.power(10, color)
        avg_loss = np.mean(loss)
        ax = f.add_subplot(19, 13, int(x / number_of_model_plot) + (x % number_of_model_plot) * 13 + 1)
        #         ax = plt.subplot(number_of_model_plot, 6 ,1+6*x)
        bins = 30
        denominator, xedges, yedges = np.histogram2d(XYL[:, 0], XYL[:, 1], bins=bins)
        nominator, _, _ = np.histogram2d(XYL[:, 0], XYL[:, 1], bins=[xedges, yedges], weights=color)

        result = nominator / denominator
        result = result.T

        X, Y = np.meshgrid(xedges, yedges)
        ax.pcolormesh(X, Y, result, cmap=cmap, vmin=minmax2[int(x / num_share)][0],
                      vmax=minmax2[int(x / num_share)][1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('white')
        ax.set_title("Avg_test_loss(log10): {:10.3e}".format(np.log10(avg_loss)))
        norm = mpl.colors.Normalize(vmin=minmax2[int(x / num_share)][0],
                                    vmax=minmax2[int(x / num_share)][1])
        f.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm))


def color2(test_path, minmax1, minmax2, num_share, number_of_model_plot, f):
    cmap = plt.cm.bwr
    for x, _ in tqdm(enumerate(test_path), desc="outer"):
        train_path = '/'.join(_.split('/')[:-1])

        XYL = pickle.load(open(os.path.join(_, 'XYL.p'), "rb"))
        num_node = 100
        color = XYL[:, 2]
        ax = f.add_subplot(1, number_of_model_plot, int(x) + 1)
        #       ax = plt.subplot(number_of_model_plot, 6 ,1+6*x)
        bins = 30
        denominator, xedges, yedges = np.histogram2d(XYL[:, 0], XYL[:, 1], bins=bins)
        nominator, _, _ = np.histogram2d(XYL[:, 0], XYL[:, 1], bins=[xedges, yedges], weights=color)

        result = nominator / denominator
        result = result.T

        X, Y = np.meshgrid(xedges, yedges)
        ax.pcolormesh(X, Y, result, cmap=cmap, vmin=minmax2[int(x / num_share)][0],
                      vmax=minmax2[int(x / num_share)][1])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('white')
        norm = mpl.colors.Normalize(vmin=minmax2[int(x / num_share)][0],
                                    vmax=minmax2[int(x / num_share)][1])
        f.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm))


def Cal_column_range(test_path, num_share_model):
    shared_column_1 = []
    shared_column_2 = []

    for x, _ in tqdm(enumerate(test_path), desc="outer"):
        XYL = pickle.load(open(os.path.join(_, 'XYL.p'), "rb"))
        color = XYL[:, 2]
        shared_column_1.append([min(color), max(color)])
        bins = 30
        denominator, xedges, yedges = np.histogram2d(XYL[:, 0], XYL[:, 1], bins=bins)
        nominator, _, _ = np.histogram2d(XYL[:, 0], XYL[:, 1], bins=[xedges, yedges], weights=color)
        result = nominator / denominator
        result = result.T
        temp = []
        for i in result:
            for j in i:
                if not np.isnan(j):
                    temp.append(j)
        shared_column_2.append([min(temp), max(temp)])

    m = np.min(np.array(shared_column_1).reshape(-1, num_share_model, 2), axis=1)
    M = np.max(np.array(shared_column_1).reshape(-1, num_share_model, 2), axis=1)
    minmax1 = np.concatenate([m[:, 0].reshape(-1, 1), M[:, 1].reshape(-1, 1)], axis=1)
    print(minmax1)

    m = np.min(np.array(shared_column_2).reshape(-1, num_share_model, 2), axis=1)
    M = np.max(np.array(shared_column_2).reshape(-1, num_share_model, 2), axis=1)
    minmax2 = np.concatenate([m[:, 0].reshape(-1, 1), M[:, 1].reshape(-1, 1)], axis=1)
    print(minmax2)

    return minmax1, minmax2


def draw_trainval_curve(test_path, minmax1, minmax2, num_share, number_of_model_plot, f):
    cmap = plt.cm.bwr
    for x, _ in tqdm(enumerate(test_path), desc="outer"):
        train_path = '/'.join(_.split('/')[:-1])
        train_config = yaml.load(open(os.path.join(train_path, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
        test_config = yaml.load(open(os.path.join(_, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
        split = test_config['dataset']['split']
        num_prop = train_config['model']['num_prop']

        train_stats = pickle.load(open(os.path.join(train_path, "train_stats.p"), "rb"))

        XYL = pickle.load(open(os.path.join(_, 'XYL.p'), "rb"))
        num_node = 100
        color = XYL[:, 2]

        ################################################################ 1

        ax = f.add_subplot(19, 13, int(x / number_of_model_plot) + (x % number_of_model_plot) * 13 + 2)

        VAL_LOSS = train_stats['val_loss']
        best_ = np.min(VAL_LOSS)
        best_epoch_ = np.where(best_ == VAL_LOSS)[0]

        ax.plot(train_stats['val_loss'], color='orange')
        ax.set_xlabel("Epoch")
        ax.scatter(best_epoch_, best_, color='red')
        #################################################################  2
        TRAIN_LOSS = train_stats['train_loss']
        best = np.min(TRAIN_LOSS)
        best_epoch = np.where(best == TRAIN_LOSS)[0]
        ax.set_ylim([-0.001, 0.1])
        ax.plot(train_stats['train_loss'], color='skyblue')
        ax.set_title(
            "Best_train_loss: {:10.2e}({}) \n Best_validation_loss : {:10.2e}({})".format(best, best_epoch, best_,
                                                                                          best_epoch_), fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.scatter(best_epoch, best, color='blue')
        ax.plot([0.1 for i in range(100)], c='red', linestyle='--')

        train_stats = 0
        best = 0
        best_epoch = 0

        ##############################################################  2.5
        try:
            ins = ax.inset_axes([0.6, 0.7, 0.4, 0.3])
            test_loss = pickle.load(open(os.path.join(_, "test_loss_list.p"), "rb"))
            ins.plot(test_loss)
        except:
            pass

        ##############################################################  2.5

        ins = ax.inset_axes([0, 0.68, 0.25, 0.32])
        bins = 30
        denominator, xedges, yedges = np.histogram2d(XYL[:, 0], XYL[:, 1], bins=bins)
        nominator, _, _ = np.histogram2d(XYL[:, 0], XYL[:, 1], bins=[xedges, yedges], weights=color)

        result = nominator / denominator
        result = result.T

        X, Y = np.meshgrid(xedges, yedges)
        ins.pcolormesh(X, Y, result, cmap=cmap, vmin=minmax2[int(x / num_share)][0],
                       vmax=minmax2[int(x / num_share)][1])

        ins.set_xticks([])
        ins.set_yticks([])
        ins.set_facecolor('black')


def draw_trainval_curve2(test_path, minmax1, minmax2, num_share, number_of_model_plot, f):
    cmap = plt.cm.bwr
    for x, _ in tqdm(enumerate(test_path), desc="outer"):
        train_path = '/'.join(_.split('/')[:-1])
        train_config = yaml.load(open(os.path.join(train_path, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
        test_config = yaml.load(open(os.path.join(_, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
        split = test_config['dataset']['split']
        num_prop = train_config['model']['num_prop']

        train_stats = pickle.load(open(os.path.join(train_path, "train_stats.p"), "rb"))

        XYL = pickle.load(open(os.path.join(_, 'XYL.p'), "rb"))
        num_node = 100
        color = XYL[:, 2]

        ################################################################ 1

        ax = f.add_subplot(19, 13, int(x / number_of_model_plot) + (x % number_of_model_plot) * 13 + 2)

        VAL_LOSS = train_stats['val_loss']
        best_ = np.min(VAL_LOSS)
        best_epoch_ = np.where(best_ == VAL_LOSS)[0]

        ax.plot(train_stats['val_loss'], color='orange')
        ax.set_xlabel("Epoch")
        ax.scatter(best_epoch_, best_, color='red')
        #################################################################  2
        TRAIN_LOSS = train_stats['train_loss']
        best = np.min(TRAIN_LOSS)
        best_epoch = np.where(best == TRAIN_LOSS)[0]
        ax.plot(train_stats['train_loss'], color='skyblue')
        ax.set_title(
            "Best_train_loss: {:10.2e}({}) \n Best_validation_loss : {:10.2e}({}) \n".format(best, best_epoch, best_,
                                                                                             best_epoch_), fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.scatter(best_epoch, best, color='blue')
        ax.plot([0.1 for i in range(100)], c='red', linestyle='--')

        train_stats = 0
        best = 0
        best_epoch = 0

        ##############################################################  2.5
        #         try:
        #             ins = ax.inset_axes([0.6, 0.7, 0.4, 0.3])
        #             test_loss = pickle.load(open(os.path.join(_, "test_loss_list.p"), "rb"))
        #             ins.plot(test_loss)
        #         except:
        #             pass

        ##############################################################  2.5

        ins = ax.inset_axes([0, 0.68, 0.25, 0.32])
        bins = 30
        denominator, xedges, yedges = np.histogram2d(XYL[:, 0], XYL[:, 1], bins=bins)
        nominator, _, _ = np.histogram2d(XYL[:, 0], XYL[:, 1], bins=[xedges, yedges], weights=color)

        result = nominator / denominator
        result = result.T

        X, Y = np.meshgrid(xedges, yedges)
        ins.pcolormesh(X, Y, result, cmap=cmap, vmin=minmax2[int(x / num_share)][0],
                       vmax=minmax2[int(x / num_share)][1])

        ins.set_xticks([])
        ins.set_yticks([])
        ins.set_facecolor('black')


def draw_best_degree_histo(test_path, minmax1, minmax2, num_share, number_of_model_plot, f, norm):
    for x, path in tqdm(enumerate(test_path)):

        count1 = 0
        count2 = 0
        XYL = np.array(pickle.load(open(os.path.join(path, "XYL.p"), "rb")))
        name = pickle.load(open(os.path.join(path, "name.p"), "rb"))

        min_L = minmax1[int(x / num_share)][0]
        max_L = minmax1[int(x / num_share)][1]

        threshold = min_L + 0.3 * (max_L - min_L)
        graph_index_toplot = np.where(XYL[:, 2] < threshold)[0]
        num_node = 100

        deg_list = []

        for index in graph_index_toplot:
            graph_name = name[index]
            num_node == 100
            for _ in os.listdir("data_temp/exp2_test_100_0.3"):
                if graph_name in _:
                    graph_path = os.path.join("data_temp/exp2_test_100_0.3", _)
                    break
            graph = pickle.load(open(graph_path, "rb"))
            J = graph['J'].todense()
            G = nx.from_numpy_array(J)
            _, deg, cnt = degree(G)
            deg_list += [deg] * cnt
            if np.mean(deg) < 50:
                count1 += 1
            else:
                count2 += 1

        ax = f.add_subplot(19, 13, int(x / number_of_model_plot) + (x % number_of_model_plot) * 13 + 3)
        a = Counter(deg_list)

        if norm == True:
            sum_count = sum(a.values())
        else:
            sum_count = 1

        ax.bar(a.keys(), [i / sum_count for i in list(a.values())], fill=True, edgecolor='blue', width=1, label='Test')
        ax.set_ylabel("Distribution", fontsize=10)
        ax.set_xlabel("Degree", fontsize=10)

        ax.axvline(12.5, color='yellow', linestyle='--')
        ax.axvline(32.5, color='yellow', linestyle='--')
        ax.axvline(72.5, color='yellow', linestyle='--')
        ax.axvline(92.5, color='yellow', linestyle='--')
        ax.axvline(52.5, color='yellow', linestyle='--')
        ax.axvline(50, color='black', linestyle='--')

        ax.set_title("Num graph : {} / {}".format(count1, count2), fontsize=10)
        ax.set_xlim(0, 100)


def draw_train_module(test_path, number_of_model_plot, f, norm):
    def func1(mod, graph):

        try:
            module0 = graph[0]
            mod[0, module0] = 0
        except:
            pass

        try:
            module1 = graph[1]
            mod[0, module1] = 1
        except:
            pass

        try:
            module2 = graph[2]
            mod[0, module2] = 2
        except:
            pass

        try:
            module3 = graph[3]
            mod[0, module3] = 3
        except:
            pass

        try:
            module4 = graph[4]
            mod[0, module4] = 4
        except:
            pass

        return mod

    deg_list1 = pickle.load(open("meta_group_1.p", "rb"))
    deg_list2 = pickle.load(open("meta_group_2.p", "rb"))
    deg_list3 = pickle.load(open("meta_group_3.p", "rb"))
    deg_list4 = pickle.load(open("meta_group_4.p", "rb"))
    deg_list5 = pickle.load(open("meta_group_5.p", "rb"))

    for idx, path in tqdm(enumerate(test_path)):
        train_path = '/'.join(path.split('/')[:-1])
        train_config = yaml.load(open(os.path.join(train_path, 'config.yaml'), 'r'), Loader=yaml.FullLoader)

        try:
            num_module = int(train_config['model']['num_module'])
            train_group = train_config['dataset']['data_path'].split('/')[-1]

            if train_group == "meta_group_all":
                train_group_list = [1, 2, 3, 4, 5]
            else:
                train_group_list = [int(___) for ___ in train_group.split("_")[2:]]
            print(train_group_list)

        except:
            num_module = 1
            train_group = None

        try:
            m_1 = pickle.load(open(os.path.join(train_path, "m1.p"), "rb"))
            m_2 = pickle.load(open(os.path.join(train_path, "m2.p"), "rb"))
            m_3 = pickle.load(open(os.path.join(train_path, "m3.p"), "rb"))
            m_4 = pickle.load(open(os.path.join(train_path, "m4.p"), "rb"))
            m_5 = pickle.load(open(os.path.join(train_path, "m5.p"), "rb"))

            print("TRAIN MODULE LOADED")
        except:
            print("GENERATING TRAIN MODULE")
            if num_module != 1:
                node_hist = os.path.join(train_path, "node_module_hist.p")
                node_hist = np.array(pickle.load(open(node_hist, "rb")))

                for idx2, i in tqdm(enumerate([-1])):
                    m_1 = []
                    m_2 = []
                    m_3 = []
                    m_4 = []
                    m_5 = []

                    one_epoch = node_hist[i]

                    group1 = []
                    group2 = []
                    group3 = []
                    group4 = []
                    group5 = []

                    if len(train_group_list) == 1:
                        for b in one_epoch[:50]:
                            for graph in b:
                                mod = np.zeros([1, 100])
                                mod = func1(mod, graph)
                                group1 += mod.tolist()[0]

                    elif len(train_group_list) == 2:
                        for b in one_epoch[:50]:
                            for graph in b:
                                mod = np.zeros([1, 100])
                                mod = func1(mod, graph)
                                group1 += mod.tolist()[0]

                        for b in one_epoch[150:200]:
                            for graph in b:
                                mod = np.zeros([1, 100])
                                mod = func1(mod, graph)
                                group2 += mod.tolist()[0]

                    elif len(train_group_list) == 3:
                        for b in one_epoch[:50]:
                            for graph in b:
                                mod = np.zeros([1, 100])
                                mod = func1(mod, graph)
                                group1 += mod.tolist()[0]

                        for b in one_epoch[150:200]:
                            for graph in b:
                                mod = np.zeros([1, 100])
                                mod = func1(mod, graph)
                                group2 += mod.tolist()[0]

                        for b in one_epoch[300:350]:
                            for graph in b:
                                mod = np.zeros([1, 100])
                                mod = func1(mod, graph)
                                group3 += mod.tolist()[0]

                    elif len(train_group_list) == 4:
                        for b in one_epoch[:50]:
                            for graph in b:
                                mod = np.zeros([1, 100])
                                mod = func1(mod, graph)
                                group1 += mod.tolist()[0]

                        for b in one_epoch[150:200]:
                            for graph in b:
                                mod = np.zeros([1, 100])
                                mod = func1(mod, graph)
                                group2 += mod.tolist()[0]

                        for b in one_epoch[300:350]:
                            for graph in b:
                                mod = np.zeros([1, 100])
                                mod = func1(mod, graph)
                                group3 += mod.tolist()[0]

                        for b in one_epoch[450:500]:
                            for graph in b:
                                mod = np.zeros([1, 100])
                                mod = func1(mod, graph)
                                group4 += mod.tolist()[0]


                    elif len(train_group_list) == 5:
                        for b in one_epoch[:50]:
                            for graph in b:
                                mod = np.zeros([1, 100])
                                mod = func1(mod, graph)
                                group1 += mod.tolist()[0]

                        for b in one_epoch[150:200]:
                            for graph in b:
                                mod = np.zeros([1, 100])
                                mod = func1(mod, graph)
                                group2 += mod.tolist()[0]

                        for b in one_epoch[300:350]:
                            for graph in b:
                                mod = np.zeros([1, 100])
                                mod = func1(mod, graph)
                                group3 += mod.tolist()[0]

                        for b in one_epoch[450:500]:
                            for graph in b:
                                mod = np.zeros([1, 100])
                                mod = func1(mod, graph)
                                group4 += mod.tolist()[0]

                        for b in one_epoch[600:650]:
                            for graph in b:
                                mod = np.zeros([1, 100])
                                mod = func1(mod, graph)
                                group5 += mod.tolist()[0]

                    GROUP = [group1, group2, group3, group4, group5]
                    for IDX, T_G in enumerate(reversed(train_group_list)):
                        if T_G == 1:
                            DEG_LIST = deg_list1
                        elif T_G == 2:
                            DEG_LIST = deg_list2
                        elif T_G == 3:
                            DEG_LIST = deg_list3
                        elif T_G == 4:
                            DEG_LIST = deg_list4
                        elif T_G == 5:
                            DEG_LIST = deg_list5

                        for d, m in zip(DEG_LIST, GROUP[IDX]):
                            if m == 0:
                                m_1.append(d)
                            elif m == 1:
                                m_2.append(d)
                            elif m == 2:
                                m_3.append(d)
                            elif m == 3:
                                m_4.append(d)
                            elif m == 4:
                                m_5.append(d)

                with open(os.path.join(train_path, "m1.p"), "wb") as a1f:
                    pickle.dump(m_1, a1f)
                with open(os.path.join(train_path, "m2.p"), "wb") as a2f:
                    pickle.dump(m_2, a2f)
                with open(os.path.join(train_path, "m3.p"), "wb") as a3f:
                    pickle.dump(m_3, a3f)
                with open(os.path.join(train_path, "m4.p"), "wb") as a4f:
                    pickle.dump(m_4, a4f)
                with open(os.path.join(train_path, "m5.p"), "wb") as a5f:
                    pickle.dump(m_5, a5f)

        a1 = Counter(m_1)
        a2 = Counter(m_2)
        a3 = Counter(m_3)
        a4 = Counter(m_4)
        a5 = Counter(m_5)

        if norm == True:
            Z = len(m_1) + len(m_2) + len(m_3) + len(m_4) + len(m_5)
        else:
            Z = 1

        ax = f.add_subplot(19, 13, int(idx / number_of_model_plot) + (idx % number_of_model_plot) * 13 + 4)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Distribution")
        ax.set_title("[{}/{}]".format(len(m_1), len(m_2)))
        ax.bar(a2.keys(), [ii / Z for ii in list(a2.values())], width=1, color='orange')
        ax.bar(a1.keys(), [ii / Z for ii in list(a1.values())], width=1, color='blue', fill=False)

        if num_module == 3:
            ax.bar(a3.keys(), [ii / Z for ii in list(a3.values())], width=1, color='green', alpha=0.5)
            ax.set_title("[{}/{}/{}]".format(len(m_1), len(m_2), len(m_3)))

        elif num_module == 4:
            ax.bar(a3.keys(), [ii / Z for ii in list(a3.values())], width=1, color='red', alpha=0.5)
            ax.bar(a4.keys(), [ii / Z for ii in list(a4.values())], width=1, color='purple', alpha=0.5)
            ax.set_title("[{}/{}/{}/{}]".format(len(m_1), len(m_2), len(m_3), len(m_4)))

        elif num_module == 5:
            ax.bar(a3.keys(), [ii / Z for ii in list(a3.values())], width=1, color='red', alpha=0.5)
            ax.bar(a4.keys(), [ii / Z for ii in list(a4.values())], width=1, color='purple', alpha=0.5)
            ax.bar(a5.keys(), [ii / Z for ii in list(a5.values())], width=1, color='yellow', alpha=0.5)
            ax.set_title("[{}/{}/{}/{}/{}]".format(len(m_1), len(m_2), len(m_3), len(m_4), len(m_5)))


def draw_test_deg_module_reject(test_path, minmax1, minmax2, num_share, number_of_model_plot, f, norm):
    for idx1, path in enumerate(test_path):

        train_path = '/'.join(path.split('/')[:-1])
        train_config = yaml.load(open(os.path.join(train_path, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
        try:
            num_module = int(train_config['model']['num_module'])
            train_group = train_config['dataset']['data_path'].split('/')[-1]
        except:
            num_module = 1

        if num_module != 1:
            a = os.path.join(path, "node_module_hist.p")
            a = np.array(pickle.load(open(a, "rb")))

            name = pickle.load(open(os.path.join(path, "file.p"), "rb"))

            XYL = np.array(pickle.load(open(os.path.join(path, "XYL.p"), "rb")))

            min_L = minmax1[int(idx1 / num_share)][0]
            max_L = minmax1[int(idx1 / num_share)][1]

            threshold = min_L + 0.3 * (max_L - min_L)
            graph_index_toplot = np.where(XYL[:, 2] < threshold)[0]

            all_deg_list = pickle.load(open("all_deg_list.p", "rb"))
            deg_list_ = np.array(all_deg_list)[graph_index_toplot].reshape(-1, )

            #############################################
            one_epoch_ = a[0]
            one_epoch = []

            for iii in one_epoch_:
                one_epoch += iii

            one_epoch = np.array(one_epoch)[graph_index_toplot].reshape(-1, )

            target_dist = defaultdict(int)
            for m, d in zip(one_epoch, deg_list_):
                target_dist[d] += 1

            Z = sum(list(target_dist.values()))

            mu = np.mean(deg_list_)
            sigma = np.std(deg_list_)

            def q(xx, mu, sigma):
                return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(xx - mu) ** 2 / (2 * sigma ** 2))

            def p(target_dist, xx):
                return target_dist[xx]

            def rejection_sampling(deg_list_, mu, sigma, iter=1000):
                samples = []
                new_deg_list = []
                new_mod_list = []
                new_idx_list = []

                for ii in tqdm(range(iter)):

                    idx = np.random.randint(0, len(deg_list_), 1)[0]
                    z = deg_list_[idx]
                    u = np.random.uniform(0, p(target_dist, z))
                    if u <= (q(z, mu, sigma) * k):
                        samples.append(z)
                        new_idx_list.append(idx)
                        new_deg_list.append(deg_list_[idx])
                        new_mod_list.append(one_epoch[idx])

                return np.array(samples), new_deg_list, new_mod_list, new_idx_list

            x = sorted(np.unique(deg_list_))
            k = 999999
            for ii in x:
                k = min(p(target_dist, ii) / q(ii, mu, sigma), k)

            s, new_deg_list, new_mod_list, new_idx_list = rejection_sampling(deg_list_, mu, sigma, iter=1000000)

            aa = Counter(new_deg_list)
            ZZ = sum(list(aa.values()))

            #############################################

            for idx2, i in tqdm(enumerate([-1])):
                one_epoch_ = a[i]
                one_epoch = []

                for iii in one_epoch_:
                    one_epoch += iii

                one_epoch = np.array(one_epoch)[graph_index_toplot].reshape(-1, )[new_idx_list]

                m_1 = []
                m_2 = []
                m_3 = []
                m_4 = []
                m_5 = []

                for d, m in zip(new_deg_list, one_epoch):
                    if m == 0:
                        m_1.append(d)
                    elif m == 1:
                        m_2.append(d)
                    elif m == 2:
                        m_3.append(d)
                    elif m == 3:
                        m_4.append(d)
                    elif m == 4:
                        m_5.append(d)

                a1 = Counter(m_1)
                a2 = Counter(m_2)
                a3 = Counter(m_3)
                a4 = Counter(m_4)
                a5 = Counter(m_5)

                if norm == True:
                    Z_ = 0
                    for __ in list(a1.values()):
                        Z_ += __
                    for __ in list(a2.values()):
                        Z_ += __
                    for __ in list(a3.values()):
                        Z_ += __
                    for __ in list(a4.values()):
                        Z_ += __
                    for __ in list(a5.values()):
                        Z_ += __
                else:
                    Z_ = 1

                ax = f.add_subplot(19, 13, int(idx1 / number_of_model_plot) + (idx1 % number_of_model_plot) * 13 + 5)
                ax.set_xlim(0, 100)

                ax.axvline(12.5, color='yellow', linestyle='--')
                ax.axvline(32.5, color='yellow', linestyle='--')
                ax.axvline(72.5, color='yellow', linestyle='--')
                ax.axvline(92.5, color='yellow', linestyle='--')
                ax.axvline(52.5, color='yellow', linestyle='--')
                ax.axvline(50, color='black', linestyle='--')
                if num_module == 2:
                    ax.bar(a2.keys(), [ii / Z_ for ii in list(a2.values())], width=1, color='orange')
                    ax.bar(a1.keys(), [ii / Z_ for ii in list(a1.values())], width=1, color='blue', fill=False)
                    ax.set_title("[{}/{}]".format(len(m_1), len(m_2)))
                if num_module == 3:
                    ax.bar(a3.keys(), [ii / Z_ for ii in list(a3.values())], width=1, color='green', alpha=0.5)
                    ax.set_title("[{}/{}/{}]".format(len(m_1), len(m_2), len(m_3)))

                elif num_module == 4:
                    ax.bar(a3.keys(), [ii / Z_ for ii in list(a3.values())], width=1, color='red', alpha=0.5)
                    ax.bar(a4.keys(), [ii / Z_ for ii in list(a4.values())], width=1, color='purple', alpha=0.5)
                    ax.set_title("[{}/{}/{}/{}]".format(len(m_1), len(m_2), len(m_3), len(m_4)))

                elif num_module == 5:
                    ax.bar(a3.keys(), [ii / Z_ for ii in list(a3.values())], width=1, color='red', alpha=0.5)
                    ax.bar(a4.keys(), [ii / Z_ for ii in list(a4.values())], width=1, color='purple', alpha=0.5)
                    ax.bar(a5.keys(), [ii / Z_ for ii in list(a5.values())], width=1, color='yellow', alpha=0.5)
                    ax.set_title("[{}/{}/{}/{}/{}]".format(len(m_1), len(m_2), len(m_3), len(m_4), len(m_5)))


def draw_test_deg_module(test_path, minmax1, minmax2, num_share, number_of_model_plot, f, norm):
    for idx1, path in tqdm(enumerate(test_path)):

        train_path = '/'.join(path.split('/')[:-1])
        train_config = yaml.load(open(os.path.join(train_path, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
        try:
            num_module = int(train_config['model']['num_module'])
            train_group = train_config['dataset']['data_path'].split('/')[-1]
        except:
            num_module = 1

        if num_module != 1:
            a = os.path.join(path, "node_module_hist.p")
            a = np.array(pickle.load(open(a, "rb")))

            name = pickle.load(open(os.path.join(path, "file.p"), "rb"))
            XYL = np.array(pickle.load(open(os.path.join(path, "XYL.p"), "rb")))

            min_L = minmax1[int(idx1 / num_share)][0]
            max_L = minmax1[int(idx1 / num_share)][1]

            threshold = min_L + 0.3 * (max_L - min_L)
            graph_index_toplot = np.where(XYL[:, 2] < threshold)[0]

            all_deg_list = pickle.load(open("all_deg_list.p", "rb"))
            deg_list_ = np.array(all_deg_list)[graph_index_toplot]

            for idx2, i in tqdm(enumerate([-1])):
                one_epoch_ = a[i]
                one_epoch = []

                for iii in one_epoch_:
                    one_epoch += iii

                m_1 = []
                m_2 = []
                m_3 = []
                m_4 = []
                m_5 = []

                for index, d in zip(graph_index_toplot, deg_list_):
                    m = one_epoch[index]

                    for m_, d_ in zip(m, d):
                        if m_ == 0:
                            m_1.append(d_)
                        elif m_ == 1:
                            m_2.append(d_)
                        elif m_ == 2:
                            m_3.append(d_)
                        elif m_ == 3:
                            m_4.append(d_)
                        elif m_ == 4:
                            m_5.append(d_)

                a1 = Counter(m_1)
                a2 = Counter(m_2)
                a3 = Counter(m_3)
                a4 = Counter(m_4)
                a5 = Counter(m_5)

                if norm == True:
                    Z = 0
                    for __ in list(a1.values()):
                        Z += __
                    for __ in list(a2.values()):
                        Z += __
                    for __ in list(a3.values()):
                        Z += __
                    for __ in list(a4.values()):
                        Z += __
                    for __ in list(a5.values()):
                        Z += __
                else:
                    Z = 1

                ax = f.add_subplot(19, 13, int(idx1 / number_of_model_plot) + (idx1 % number_of_model_plot) * 13 + 6)

                ax.set_xlim(0, 100)
                ax.set_xlabel("Degree")
                ax.set_ylabel("Distribution")

                ax.axvline(12.5, color='yellow', linestyle='--')
                ax.axvline(32.5, color='yellow', linestyle='--')
                ax.axvline(72.5, color='yellow', linestyle='--')
                ax.axvline(92.5, color='yellow', linestyle='--')
                ax.axvline(52.5, color='yellow', linestyle='--')
                ax.axvline(50, color='black', linestyle='--')

                ax.set_title("[{}/{}]".format(len(m_1), len(m_2)), fontsize=10)
                ax.bar(a2.keys(), [ii / Z for ii in list(a2.values())], width=1, color='orange')
                ax.bar(a1.keys(), [ii / Z for ii in list(a1.values())], width=1, color='blue', fill=False)

                if num_module == 3:
                    ax.bar(a3.keys(), [ii / Z for ii in list(a3.values())], width=1, color='green', alpha=0.5)
                    ax.set_title("[{}/{}/{}]".format(len(m_1), len(m_2), len(m_3)), fontsize=10)

                elif num_module == 4:
                    ax.bar(a3.keys(), [ii / Z for ii in list(a3.values())], width=1, color='red', alpha=0.5)
                    ax.bar(a4.keys(), [ii / Z for ii in list(a4.values())], width=1, color='purple', alpha=0.5)
                    ax.set_title("[{}/{}/{}/{}]".format(len(m_1), len(m_2), len(m_3), len(m_4)), fontsize=10)

                elif num_module == 5:
                    ax.bar(a3.keys(), [ii / Z for ii in list(a3.values())], width=1, color='red', alpha=0.5)
                    ax.bar(a4.keys(), [ii / Z for ii in list(a4.values())], width=1, color='purple', alpha=0.5)
                    ax.bar(a5.keys(), [ii / Z for ii in list(a5.values())], width=1, color='yellow', alpha=0.5)
                    ax.set_title("[{}/{}/{}/{}/{}]".format(len(m_1), len(m_2), len(m_3), len(m_4), len(m_5)),
                                 fontsize=10)


def Cal_XYL(test_path, node_only=False, graph_only=False):
    X_low100 = np.array(pickle.load(open("data_temp/WL_flex_graphs_100_shell_pca_sub_50.p", "rb"))['X_low'])
    data100 = np.array(pickle.load(open("data_temp/WL_flex_graphs_100_shell_pca_sub_50.p", "rb"))['prop_list'])
    name_list100 = np.array(pickle.load(open("data_temp/WL_flex_graphs_100_shell_pca_sub_50.p", "rb"))['name_list'])

    for _ in test_path:
        test_config = yaml.load(open(os.path.join(_, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
        split = test_config['dataset']['split']
        name = pickle.load(open(os.path.join(_, "name.p"), "rb"))
        try:
            if node_only:
                p_n = pd.read_csv(os.path.join(_, "gt_pts_n{}.csv".format(split)), sep='\t', header=None).values
                q_n = pd.read_csv(os.path.join(_, "pred_pts_n{}.csv".format(split)), sep='\t', header=None).values
            elif graph_only:
                p_g = pd.read_csv(os.path.join(_, "gt_pts_g{}.csv".format(split)), sep='\t', header=None).values
                q_g = pd.read_csv(os.path.join(_, "pred_pts_g{}.csv".format(split)), sep='\t', header=None).values
            else:
                p_n = pd.read_csv(os.path.join(_, "gt_pts_n{}.csv".format(split)), sep='\t', header=None).values
                q_n = pd.read_csv(os.path.join(_, "pred_pts_n{}.csv".format(split)), sep='\t', header=None).values
                p_g = pd.read_csv(os.path.join(_, "gt_pts_g{}.csv".format(split)), sep='\t', header=None).values
                q_g = pd.read_csv(os.path.join(_, "pred_pts_g{}.csv".format(split)), sep='\t', header=None).values
        except:
            p = pd.read_csv(os.path.join(_, "gt_pts.csv"), sep='\t', header=None).values
            q = pd.read_csv(os.path.join(_, "pred_pts.csv"), sep='\t', header=None).values
        loss_func = nn.MSELoss(reduction='mean')
        XYL = []
        G2loss = []
        G4loss = []
        with torch.no_grad():
            graph_num = 1584
            for idx in tqdm(range(graph_num), leave=False):
                if node_only:
                    P_n = torch.Tensor(p_n)[idx * 100: (idx + 1) * 100]
                    Q_n = torch.Tensor(q_n)[idx * 100: (idx + 1) * 100]
                    loss = loss_func(Q_n, P_n)
                elif graph_only:
                    P_g = torch.Tensor(p_g)[idx]
                    Q_g = torch.Tensor(q_g)[idx]
                    loss = loss_func(Q_g, P_g)
                else:
                    P_n = torch.Tensor(p_n)[idx * 100: (idx + 1) * 100]
                    Q_n = torch.Tensor(q_n)[idx * 100: (idx + 1) * 100]
                    loss_n = loss_func(Q_n, P_n)
                    P_g = torch.Tensor(p_g)[idx]
                    Q_g = torch.Tensor(q_g)[idx]
                    loss_g = loss_func(Q_g, P_g)
                    loss = (loss_n + loss_g) / 2
#                     print(loss)
                graph_name = name[idx]
                g_list = ['WL_graph_nn100_k33.46488294314381_p0.75614_0000035.p',
                          'WL_graph_nn100_k32.50167224080268_p0.42098_0000034.p',
                          'WL_graph_nn100_k72.31438127090301_p0.27924_0000038.p',
                          'WL_graph_nn100_k73.59866220735786_p0.00219_0000002.p']
                if (data100[idx, 0] >= 70) & (data100[idx, 0] <= 75):
                    G4loss.append(loss)
                elif (data100[idx, 0] >= 30) & (data100[idx, 0] <= 35):
                    G2loss.append(loss)
                
                X = -X_low100[np.where(name_list100 == graph_name)[0][0], 0].item()
                Y = -X_low100[np.where(name_list100 == graph_name)[0][0], 1].item()
    # math.log10(loss)
                XYL += [[X, Y, np.log10(loss)]]
        G4loss = np.array(G4loss)
        G2loss = np.array(G2loss)
        # print("G2 loss: ", np.mean(G2loss))
        # print("G4 loss: ", np.mean(G4loss))
        XYL = np.array(XYL)
        with open(os.path.join(_, 'XYL.p'), 'wb') as ff:
            pickle.dump(XYL, ff)
