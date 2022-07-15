from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.nn import ModuleList, ReLU, Sequential, GRUCell
from torch_scatter import scatter

from torch_geometric.nn.conv import MessagePassing
from .linear import Linear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import degree
from torch_geometric.nn.inits import reset


class PNAConv_meta(MessagePassing):
    r"""The Principal Neighbourhood Aggregation graph convolution operator
    from the `"Principal Neighbourhood Aggregation for Graph Nets"
    <https://arxiv.org/abs/2004.05718>`_ paper

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left(
        \mathbf{x}_i, \underset{j \in \mathcal{N}(i)}{\bigoplus}
        h_{\mathbf{\Theta}} \left( \mathbf{x}_i, \mathbf{x}_j \right)
        \right)

    with

    .. math::
        \bigoplus = \underbrace{\begin{bmatrix}
            1 \\
            S(\mathbf{D}, \alpha=1) \\
            S(\mathbf{D}, \alpha=-1)
        \end{bmatrix} }_{\text{scalers}}
        \otimes \underbrace{\begin{bmatrix}
            \mu \\
            \sigma \\
            \max \\
            \min
        \end{bmatrix}}_{\text{aggregators}},

    where :math:`\gamma_{\mathbf{\Theta}}` and :math:`h_{\mathbf{\Theta}}`
    denote MLPs.

    .. note::

        For an example of using :obj:`PNAConv`, see `examples/pna.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/
        examples/pna.py>`_.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        aggregators (list of str): Set of aggregation function identifiers,
            namely :obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"var"` and :obj:`"std"`.
        scalers (list of str): Set of scaling function identifiers, namely
            :obj:`"identity"`, :obj:`"amplification"`,
            :obj:`"attenuation"`, :obj:`"linear"` and
            :obj:`"inverse_linear"`.
        deg (Tensor): Histogram of in-degrees of nodes in the training set,
            used by scalers to normalize.
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default :obj:`None`)
        towers (int, optional): Number of towers (default: :obj:`1`).
        pre_layers (int, optional): Number of transformation layers before
            aggregation (default: :obj:`1`).
        post_layers (int, optional): Number of transformation layers after
            aggregation (default: :obj:`1`).
        divide_input (bool, optional): Whether the input features should
            be split between towers or not (default: :obj:`False`).
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(self, in_channels: int, out_channels: int,
                 aggregators: List[str], scalers: List[str], deg: Tensor,
                 edge_dim: Optional[int] = None, towers: int = 1,
                 pre_layers: int = 1, post_layers: int = 1,
                 divide_input: bool = False, num_module=2,**kwargs):

        kwargs.setdefault('aggr', None)
        super().__init__(node_dim=0, **kwargs)

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = aggregators
        self.scalers = scalers
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers

        deg = deg.to(torch.float)
        num_nodes = int(deg.sum())
        bin_degrees = torch.arange(deg.numel()).to(torch.float)
        self.avg_deg: Dict[str, float] = {
            'lin': float((bin_degrees * deg).sum()) / num_nodes,
            'log': float(((bin_degrees + 1.0).log() * deg).sum()) / num_nodes,
            'exp': float((bin_degrees.exp() * deg).sum()) / num_nodes,
        }

        if self.edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, self.F_in)

        self.pre_nns = ModuleList()
        self.module_1 = ModuleList()
        self.module_2 = ModuleList()
        self.gru1 = ModuleList()
        self.gru2 = ModuleList()
        for _ in range(towers):
            modules = [Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_in, self.F_in)]
            self.pre_nns.append(Sequential(*modules))

            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules_1 = [Linear(in_channels, self.F_out)]
            grus_1 = GRUCell(input_size=self.F_in, hidden_size=self.F_out)
            modules_2 = [Linear(in_channels, self.F_out)]
            grus_2 = GRUCell(input_size=self.F_in, hidden_size=self.F_out)
            
            self.module_1.append(Sequential(*modules_1))
            self.module_2.append(Sequential(*modules_2))
            self.gru1.append(grus_1)
            self.gru2.append(grus_2)

        self.lin = Linear(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if self.edge_dim is not None:
            self.edge_encoder.reset_parameters()
        for nn in self.pre_nns:
            reset(nn)
        for nn in self.module_1:
            reset(nn)
        for nn in self.module_2:
            reset(nn)
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, node_idx=None,
                node_idx_inv=None, b: Tensor = None) -> Tensor:
        """"""

        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        # propagate_type: (x: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                             node_idx=node_idx, node_idx_inv=node_idx_inv, size=None, state_prev=x)

        return self.lin(out)

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_attr: OptTensor, b: Tensor = None) -> Tensor:

        h: Tensor = x_i  # Dummy.
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in)
            edge_attr = edge_attr.repeat(1, self.towers, 1)
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        return torch.stack(hs, dim=1)

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:

        outs = []
        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            elif aggregator == 'mean':
                out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            elif aggregator == 'min':
                out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'max':
                out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'var' or aggregator == 'std':
                mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, 0, None,
                                       dim_size, reduce='mean')
                out = mean_squares - mean * mean
                if aggregator == 'std':
                    out = torch.sqrt(torch.relu(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')
            outs.append(out)
        out = torch.cat(outs, dim=-1)

        deg = degree(index, dim_size, dtype=inputs.dtype)
        deg = deg.clamp_(1).view(-1, 1, 1)

        outs = []
        for scaler in self.scalers:
            if scaler == 'identity':
                pass
            elif scaler == 'amplification':
                out = out * (torch.log(deg + 1) / self.avg_deg['log'])
            elif scaler == 'attenuation':
                out = out * (self.avg_deg['log'] / torch.log(deg + 1))
            elif scaler == 'linear':
                out = out * (deg / self.avg_deg['lin'])
            elif scaler == 'inverse_linear':
                out = out * (self.avg_deg['lin'] / deg)
            else:
                raise ValueError(f'Unknown scaler "{scaler}".')
            outs.append(out)
        return torch.cat(outs, dim=-1)
    
    def update(self, msg_agg, state_prev, node_idx, node_idx_inv):
        out = torch.zeros(msg_agg.size(0), self.F_out).cuda()
#         print(len(node_idx[0]), len(node_idx[1]))
        for i in range(len(node_idx)):
            if len(node_idx[i]) == 0: continue
            msg = msg_agg[node_idx[i], :, :]
            state = state_prev[node_idx[i], :, :]
            if i == 0:
                tmp = torch.cat([msg, state], dim=-1)
                outs = [nn[1](state[:, j], nn[0](tmp[:, j])) for j, nn in enumerate(zip(self.module_1, self.gru1))]
                aux = torch.cat(outs, dim=1)
                
            elif i == 1:
                tmp = torch.cat([msg, state], dim=-1)
                outs = [nn[1]( state[:, j], nn[0](tmp[:, j])) for j, nn in enumerate(zip(self.module_2, self.gru2))]
                aux = torch.cat(outs, dim=1)
            out[node_idx[i], :] = aux
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, towers={self.towers}, '
                f'edge_dim={self.edge_dim})')
