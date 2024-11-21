import math
import typing
from typing import Optional, Tuple, Union
import opt_einsum as oe
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_max, scatter_add
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptTensor,
    PairTensor,
    SparseTensor,
)
from torch_geometric.utils import softmax
import torch.nn as nn
if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload


def pyg_softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
            scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out


class GRITSparseConv(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            beta: bool = False,
            dropout: float = 0.,
            clamp: float = 5.0,
            edge_dim: Optional[int] = None,
            bias: bool = True,
            root_weight: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.clamp = clamp
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None
        self.act = torch.nn.ReLU()

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels, bias = True)
        self.lin_value = Linear(in_channels[0], heads * out_channels)

        nn.init.xavier_normal_(self.lin_key.weight)
        nn.init.xavier_normal_(self.lin_query.weight)
        nn.init.xavier_normal_(self.lin_value.weight)

        if edge_dim is not None:
            self.lin_edge_1 = Linear(edge_dim, heads * out_channels, bias=True)
            self.lin_edge_2 = Linear(edge_dim, heads * out_channels, bias=True)



        else:
            self.lin_edge_1 = self.register_parameter('lin_edge_1', None)
            self.lin_edge_2 = self.register_parameter('lin_edge_2', None)

        nn.init.xavier_normal_(self.lin_edge_1.weight)
        nn.init.xavier_normal_(self.lin_edge_2.weight)

        self.W_A = torch.nn.Parameter(torch.zeros(out_channels, self.heads, 1), requires_grad=True)
        self.W_EV = torch.nn.Parameter(torch.zeros(out_channels, self.heads, 1), requires_grad=True)

        nn.init.xavier_normal_(self.W_A)
        nn.init.xavier_normal_(self.W_EV)



        # self.reset_parameters()

    def reset_parameters(self):
        # super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge_1.reset_parameters()
            self.lin_edge_2.reset_parameters()

    @overload
    def forward(
            self,
            x: Union[Tensor, PairTensor],
            edge_index: Adj,
            edge_attr: OptTensor = None,
            return_attention_weights: NoneType = None,
    ) -> Tensor:
        pass

    @overload
    def forward(  # noqa: F811
            self,
            x: Union[Tensor, PairTensor],
            edge_index: Tensor,
            edge_attr: OptTensor = None,
            return_attention_weights: bool = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass

    @overload
    def forward(  # noqa: F811
            self,
            x: Union[Tensor, PairTensor],
            edge_index: SparseTensor,
            edge_attr: OptTensor = None,
            return_attention_weights: bool = None,
    ) -> Tuple[Tensor, SparseTensor]:
        pass

    def forward(  # noqa: F811
            self,
            x: Union[Tensor, PairTensor],
            edge_index: Adj,
            edge_attr: OptTensor = None,
            return_attention_weights: Optional[bool] = None,
    ) -> Union[
        Tensor,
        Tuple[Tensor, Tuple[Tensor, Tensor]],
        Tuple[Tensor, SparseTensor],
    ]:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor,
        #                  edge_attr: OptTensor)
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr)

        alpha = self._alpha

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.sum(dim=1)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out, alpha

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:



        W_ew_eij = self.lin_edge_1(edge_attr).view(-1, self.heads,
                                                   self.out_channels)
        # W_eb * e_ij
        W_eb_eij = self.lin_edge_2(edge_attr).view(-1, self.heads,
                                                   self.out_channels)

        # GRIT 1. update edge_attr
        # e_ij_new = relu(\rho((query_i +key_j) * W_ew_eij) + W_eb_eij)

        score = (query_i + key_j) * W_ew_eij
        score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
        score = score + W_eb_eij

        score = self.act(score)
        edge_attr = score
        self._alpha = edge_attr.flatten(1)

        score_1 = oe.contract("ehd, dhc->ehc", score, self.W_A, backend="torch")
        # score_1 = score

        if self.clamp is not None:
            score_1 = torch.clamp(score_1, min=-self.clamp, max=self.clamp)
        score_1 = softmax(score_1, index, ptr, size_i)
        # 2. Sparse attention
        score_1 = F.dropout(score_1, p=self.dropout, training=self.training)

        score_2 = oe.contract("ehd, dhc->ehc", score, self.W_EV, backend="torch")
        # score_2 = score
        # 3. Sparse aggregation by neighbourhoods
        out = value_j + score_2
        out = out * score_1.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')