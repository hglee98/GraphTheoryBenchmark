import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from model.layers import Set2Set
from torch.nn import Embedding
from model.meta_layer import GATConv_meta, GATConv_V2
from .gat_conv import GATConv

EPS = float(np.finfo(np.float32).eps)

__all__ = ['GatGnn']


def get_weighted_average(loss_node, loss_graph, output):
    return (loss_node * output[0].shape[-1] + loss_graph * output[1].shape[-1]) / (
            output[0].shape[-1] + output[1].shape[-1])


class GatGnn(nn.Module):
    def __init__(self, config, test=False):
        """ Multitask meta gnn """
        super(GatGnn, self).__init__()
        self.config = config

        self.include_b = config.model.include_b
        self.drop_prob = config.model.drop_prob
        self.node_only = config.model.node_only
        self.graph_only = config.model.graph_only
        self.num_node = config.dataset.num_node
        self.hidden_dim = config.model.hidden_dim
        self.num_prop = config.model.num_prop
        self.aggregate_type = config.model.aggregate_type
        self.degree_emb = config.model.degree_emb
        self.jumping = config.model.jumping
        self.training = not test
        self.skip_connection = config.model.skip_connection
        self.interpol = config.model.interpol
        self.master_node = config.model.master_node if config.model.master_node is not None else False
        self.batch_size = config.test.batch_size if test else config.train.batch_size
        self.masking = config.model.masking if config.model.masking is not None else False
        self.SSL = config.model.SSL if config.model.SSL is not None else False
        self.train_pretext = config.model.train_pretext if config.model.train_pretext is not None else False
        self.num_module = config.model.num_module
        assert self.aggregate_type in ['att', 'mean', 'add', 'att_max'], 'not implemented'

        # PROPAGATION LAYER
        if not self.include_b:
            print("Not Include node attribute")
            if self.aggregate_type == 'att_max':
                self.GeoLayer1 = GATConv(2, self.hidden_dim, agg='max')
                self.GeoLayer = GATConv(self.hidden_dim, self.hidden_dim, agg='max')
            elif self.aggregate_type == 'att':
                self.GeoLayer1 = GATConv(2, self.hidden_dim)
                self.GeoLayer = GATConv(self.hidden_dim, self.hidden_dim)
            self.gru1 = nn.GRUCell(input_size=2, hidden_size=self.hidden_dim)
            self.gru = nn.GRUCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        else:
            print("Include node attribute")
            if self.aggregate_type == 'att_max':
                self.GeoLayer = GATConv(self.hidden_dim, self.hidden_dim, agg='max')
            else:
                self.GeoLayer = GATConv(self.hidden_dim, self.hidden_dim)
            self.gru = nn.GRUCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)

        # OUTPUT FUNCTION
        if self.include_b:
            node_input_dim = self.hidden_dim + 2
            graph_input_dim = self.hidden_dim * 2 + 4
            set2set_input_dim = self.hidden_dim + 2
        else:
            node_input_dim = self.hidden_dim
            graph_input_dim = 2 * self.hidden_dim
            set2set_input_dim = self.hidden_dim
        if self.node_only:
            self.node_read_out = nn.Sequential(*[
                nn.Linear(node_input_dim, 16),
                nn.LeakyReLU(),
                nn.Linear(16, 8),
                nn.LeakyReLU(),
                nn.Linear(8, 3),
                nn.LeakyReLU()
            ])
        elif self.graph_only:
            self.graph_read_out = nn.Sequential(*[
                nn.Linear(graph_input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 3),
                nn.LeakyReLU()
            ])
            self.s2s = Set2Set(set2set_input_dim, self.num_node)
        else:
            self.node_read_out = nn.Sequential(*[
                nn.Linear(node_input_dim, 16),
                nn.LeakyReLU(),
                nn.Linear(16, 8),
                nn.LeakyReLU(),
                nn.Linear(8, 3),
                nn.LeakyReLU()
            ])
            self.graph_read_out = nn.Sequential(*[
                nn.Linear(graph_input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 3),
                nn.LeakyReLU()
            ])
            self.s2s = Set2Set(set2set_input_dim, self.num_node)
        self.loss_func = nn.MSELoss(reduction='mean')
        self._init_param()

    def _init_param(self):
        if self.node_only:
            mlp_modules = [
                xx for xx in [self.node_read_out] if xx is not None
            ]
        elif self.graph_only:
            mlp_modules = [
                xx for xx in [self.graph_read_out] if xx is not None
            ]
        else:
            mlp_modules = [
                xx for xx in [self.node_read_out, self.graph_read_out] if xx is not None
            ]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, edge_index=None, target_n=None, target_g=None, batch=None):
        """
          edge_index: shape 2 X |E|
          x: shape |V| X 1
          target: shape (|V| x 3, |g| x 3)
        """
        if not self.include_b:
            state = self.GeoLayer1(x, edge_index)
            state = self.gru1(x, state)
        else:
            num_node = x.shape[0]
            state = torch.zeros(num_node, self.hidden_dim).to(x.device)
        # propagation
        for _ in range(self.num_prop - 1):
            if not self.include_b:
                Y = self.GeoLayer(state, edge_index)
            else:
                Y = self.GeoLayer(state, edge_index, b=x)
            state = self.gru(state, Y)
        target_g = target_g.reshape(-1, 3)
        target_n = target_n.reshape(-1, 3)
        if self.include_b:
            state = torch.cat([state, x], dim=1)
        if self.node_only:
            y_node = self.node_read_out(state)
            out = y_node
            loss = self.loss_func(out, target_n)
            out_ = out.view(-1, self.num_node, 3)
        elif self.graph_only:
            y_graph = self.s2s(state, batch)
            y_graph = self.graph_read_out(y_graph)
            out = y_graph
            loss = self.loss_func(out, target_g)
            out_ = out.view(-1, 3)
        else:
            y_node = self.node_read_out(state)
            y_graph = self.s2s(state, batch)
            y_graph = self.graph_read_out(y_graph)
            out = (y_node, y_graph)
            nodes_loss = self.loss_func(out[0], target_n)
            graph_loss = self.loss_func(out[1], target_g)
            weighted_average = get_weighted_average(nodes_loss, graph_loss, out)
            loss = weighted_average

        
        return out, loss


