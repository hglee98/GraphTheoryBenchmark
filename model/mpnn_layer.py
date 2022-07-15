import torch
import torch.nn as nn

from typing import Union, Tuple, Optional
from torch import Tensor
from torch.nn import Sequential as Seq, Linear, ReLU
from .MessagePassing import MessagePassing
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
)


class mpnn(MessagePassing):

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            hidden_dim: int,
            num_module: int = 2,
            aggregate_type: str = 'max',
            **kwargs,
    ):
        super().__init__(aggr=aggregate_type)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_module = num_module
        self.msg_func = Seq(Linear(2 * in_channels, out_channels))
        self.post_layer1 = Seq(Linear(out_channels + in_channels, out_channels))
        # update function
        self.update_func1 = nn.GRUCell(
            input_size=self.in_channels, hidden_size=self.hidden_dim)
        if num_module == 2:
            self.update_func2 = nn.GRUCell(
                input_size=self.in_channels, hidden_size=self.hidden_dim)
            self.post_layer2 = Seq(Linear(out_channels + in_channels, out_channels))
        if num_module == 3:
            self.update_func2 = nn.GRUCell(
                input_size=self.in_channels, hidden_size=self.hidden_dim)
            self.update_func3 = nn.GRUCell(
                input_size=self.in_channels, hidden_size=self.hidden_dim)
        elif num_module == 5:
            self.update_func2 = nn.GRUCell(
                input_size=self.in_channels, hidden_size=self.hidden_dim)
            self.update_func3 = nn.GRUCell(
                input_size=self.in_channels, hidden_size=self.hidden_dim)
            self.update_func4 = nn.GRUCell(
                input_size=self.in_channels, hidden_size=self.hidden_dim)
            self.update_func5 = nn.GRUCell(
                input_size=self.in_channels, hidden_size=self.hidden_dim)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, node_idx=None, node_idx_inv=None, edge_idx=None,
                edge_idx_inv=None, b=None):
        out = self.propagate(edge_index, x=x, state_prev=x, node_idx=node_idx
                             , node_idx_inv=node_idx_inv)
        return out

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        temp = torch.cat([x_i, x_j], dim=-1)
        return self.msg_func(temp)

    def update(self, msg_agg, state_prev, node_idx, node_idx_inv):
        out = torch.zeros(msg_agg.size(0), self.hidden_dim).cuda()
        for i in range(len(node_idx)):
            if len(node_idx[i]) == 0: continue
            msg = msg_agg[node_idx[i], :]
            state = state_prev[node_idx[i], :]
            if i == 0:
                tmp = torch.cat([msg, state], dim=-1)
                tmp = self.post_layer1(tmp)
                aux = self.update_func1(state, tmp)
            elif i == 1:
                tmp = torch.cat([msg, state], dim=-1)
                tmp = self.post_layer2(tmp)
                aux = self.update_func2(state, tmp)

            out[node_idx[i], :] = aux
        return out


class mpnn_non(MessagePassing):

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            hidden_dim: int,
            aggregate_type: str='max',
            **kwargs,
    ):
        super().__init__(aggr=aggregate_type)
        self.msg_func = Seq(Linear(2 * in_channels, out_channels))
        self.post_layer = Seq(Linear(out_channels+in_channels, out_channels))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, b=None):
        out = self.propagate(edge_index, x=x)
        out = torch.cat([x, out], dim=-1)
        return self.post_layer(out)

    def message(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.msg_func(tmp)
