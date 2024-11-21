from genericpath import exists
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsnap import batch
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax

from .GOAT.vq import VectorQuantizerEMA
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch_geometric.graphgym.register import *

@register_layer('GOAT')
class TransformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            spatial_size: int,
            global_dim: int,
            heads: int = 1,
            concat: bool = True,
            beta: bool = False,
            dropout: float = 0.,
            edge_dim: Optional[int] = None,
            bias: bool = True,
            skip: bool = True,
            dist_count_norm: bool = True,
            conv_type: str = 'local',
            num_centroids: Optional[int] = None,
            # centroid_dim: int = 64,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and skip
        self.skip = skip
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.spatial_size = spatial_size
        self.dist_count_norm = dist_count_norm
        self.conv_type = conv_type
        self.num_centroids = num_centroids
        self._alpha = None

        self.lin_key = Linear(in_channels, heads * out_channels)
        self.lin_query = Linear(in_channels, heads * out_channels)
        self.lin_value = Linear(in_channels, heads * out_channels)
        # if edge_dim is not None:
        #     self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        # else:
        #     self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels, heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels, out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)



        self.spatial_encoder = torch.nn.Embedding(spatial_size, heads)

        if self.conv_type != 'local' :
            self.vq = VectorQuantizerEMA(
                num_centroids,
                global_dim,
                decay=0.99
            )

            self.attn_fn = F.softmax

            self.lin_proj_g = Linear(in_channels, global_dim)
            self.pre_trans = Linear(in_channels, global_dim*2)
            self.lin_key_g = Linear(global_dim*2, heads * out_channels)
            self.lin_query_g = Linear(global_dim*2, heads * out_channels)
            self.lin_value_g = Linear(global_dim, heads * out_channels)



        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        # if self.edge_dim:
        #     self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

        torch.nn.init.zeros_(self.spatial_encoder.weight)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_idx = batch.batch

        if self.conv_type == 'local':
            out = self.local_forward(x, edge_index, edge_attr)[:len(batch_idx)]

        elif self.conv_type == 'global':
            out = self.global_forward(x[:len(batch_idx)], batch_idx, batch.num_nodes)

        elif self.conv_type == 'full':
            out_local = self.local_forward(x, edge_index, edge_attr)[:len(batch_idx)]
            out_global = self.global_forward(x[:len(batch_idx)], batch_idx, batch.num_nodes)
            out = torch.cat([out_local, out_global], dim=1)

        else:
            raise NotImplementedError

        batch.x = out

        return batch

    def global_forward(self, x, batch_idx, num_nodes):
        c = torch.randint(0, 64, (num_nodes,), dtype=torch.short).to(x.device)
        self.register_buffer('c_idx', c)


        d, h = self.out_channels, self.heads
        scale = 1.0 / math.sqrt(d)


        q_x = self.lin_proj_g(x)
        q_x =  self.pre_trans(q_x)
        k_x = self.vq.get_k()
        v_x = self.vq.get_v()

        q = self.lin_query_g(q_x)
        k = self.lin_key_g(k_x)
        v = self.lin_value_g(v_x)

        q, k, v = map(lambda t: rearrange(t, 'n (h d) -> h n d', h=h), (q, k, v))
        dots = torch.einsum('h i d, h j d -> h i j', q, k) * scale

        c, c_count = self.c_idx.unique(return_counts=True)
        # print(f'c count mean:{c_count.float().mean().item()}, min:{c_count.min().item()}, max:{c_count.max().item()}')

        centroid_count = torch.zeros(self.num_centroids, dtype=torch.long).to(x.device)
        centroid_count[c.to(torch.long)] = c_count.to(x.device)
        dots += torch.log(centroid_count.view(1, 1, -1))

        attn = self.attn_fn(dots, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.einsum('h i j, h j d -> h i d', attn, v)
        out = rearrange(out, 'h n d -> n (h d)')

        # Update the centroids
        if self.training:
            x_idx = self.vq.update(q_x)

            self.c_idx[batch_idx] = x_idx.squeeze().to(dtype=torch.short, device=x.device)

        return out

    def local_forward(self, x: Tensor, edge_index: Adj,
                      edge_attr: OptTensor = None):

        H, C = self.heads, self.out_channels

        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)
        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)

        if self.skip:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        # if self.lin_edge is not None:
        #     assert edge_attr is not None
        #     edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        #     key_j += edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        edge_dist, edge_dist_count = edge_attr[0].unsqueeze(1), edge_attr[1]

        # print(edge_dist)
        alpha += self.spatial_encoder(edge_dist.long())
        print(alpha)
        if self.dist_count_norm:
            alpha -= torch.log(edge_dist_count).unsqueeze_(1)

        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

# batch norm