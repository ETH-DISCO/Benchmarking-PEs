import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)
from torch_scatter import scatter
import torch_sparse
"""
Generic Node and Edge encoders for datasets with node/edge features that
consist of only one type dictionary thus require a single nn.Embedding layer.

The number of possible Node and Edge types must be set by cfg options:
1) cfg.dataset.node_encoder_num_types
2) cfg.dataset.edge_encoder_num_types

In case of a more complex feature set, use a data-specific encoder.

These generic encoders can be used e.g. for:
* ZINC
cfg.dataset.node_encoder_num_types: 28
cfg.dataset.edge_encoder_num_types: 4

* AQSOL
cfg.dataset.node_encoder_num_types: 65
cfg.dataset.edge_encoder_num_types: 5


=== Description of the ZINC dataset === 
https://github.com/graphdeeplearning/benchmarking-gnns/issues/42
The node labels are atom types and the edge labels atom bond types.

Node labels:
'C': 0
'O': 1
'N': 2
'F': 3
'C H1': 4
'S': 5
'Cl': 6
'O -': 7
'N H1 +': 8
'Br': 9
'N H3 +': 10
'N H2 +': 11
'N +': 12
'N -': 13
'S -': 14
'I': 15
'P': 16
'O H1 +': 17
'N H1 -': 18
'O +': 19
'S +': 20
'P H1': 21
'P H2': 22
'C H2 -': 23
'P +': 24
'S H1 +': 25
'C H1 -': 26
'P H1 +': 27

Edge labels:
'NONE': 0
'SINGLE': 1
'DOUBLE': 2
'TRIPLE': 3


=== Description of the AQSOL dataset === 
Node labels: 
'Br': 0, 'C': 1, 'N': 2, 'O': 3, 'Cl': 4, 'Zn': 5, 'F': 6, 'P': 7, 'S': 8, 'Na': 9, 'Al': 10,
'Si': 11, 'Mo': 12, 'Ca': 13, 'W': 14, 'Pb': 15, 'B': 16, 'V': 17, 'Co': 18, 'Mg': 19, 'Bi': 20, 'Fe': 21,
'Ba': 22, 'K': 23, 'Ti': 24, 'Sn': 25, 'Cd': 26, 'I': 27, 'Re': 28, 'Sr': 29, 'H': 30, 'Cu': 31, 'Ni': 32,
'Lu': 33, 'Pr': 34, 'Te': 35, 'Ce': 36, 'Nd': 37, 'Gd': 38, 'Zr': 39, 'Mn': 40, 'As': 41, 'Hg': 42, 'Sb':
43, 'Cr': 44, 'Se': 45, 'La': 46, 'Dy': 47, 'Y': 48, 'Pd': 49, 'Ag': 50, 'In': 51, 'Li': 52, 'Rh': 53,
'Nb': 54, 'Hf': 55, 'Cs': 56, 'Ru': 57, 'Au': 58, 'Sm': 59, 'Ta': 60, 'Pt': 61, 'Ir': 62, 'Be': 63, 'Ge': 64
    
Edge labels: 
'NONE': 0, 'SINGLE': 1, 'DOUBLE': 2, 'AROMATIC': 3, 'TRIPLE': 4
"""
def full_edge_index(edge_index, batch=None):
    """
    Retunr the Full batched sparse adjacency matrices given by edge indices.
    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.
    Implementation inspired by `torch_geometric.utils.to_dense_adj`
    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.
        batch: Batch vector, which assigns each node to a specific example.
    Returns:
        Complementary edge index.
    """

    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch,
                        dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short,
                         device=edge_index.device)

        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        # _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_full = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_full

@register_node_encoder('TypeDictNode')
class TypeDictNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        num_types = cfg.dataset.node_encoder_num_types
        if num_types < 1:
            raise ValueError(f"Invalid 'node_encoder_num_types': {num_types}")

        self.encoder = torch.nn.Embedding(num_embeddings=num_types,
                                          embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        # Encode just the first dimension if more exist
        batch.x = self.encoder(batch.x[:, 0])

        return batch


@register_edge_encoder('TypeDictEdge')
class TypeDictEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        num_types = cfg.dataset.edge_encoder_num_types
        if num_types < 1:
            raise ValueError(f"Invalid 'edge_encoder_num_types': {num_types}")

        self.encoder = torch.nn.Embedding(num_embeddings=num_types,
                                          embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)
        self.pad_to_full_graph = cfg.get("pad_to_full_graph", False)
        self.fill_value = 0.
        padding = torch.ones(1, emb_dim, dtype=torch.float) * self.fill_value
        self.register_buffer("padding", padding)
    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr)
        out_idx = batch.edge_index
        out_val = batch.edge_attr
        if self.pad_to_full_graph:
            edge_index_full = full_edge_index(out_idx, batch=batch.batch)
            edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
            # zero padding to fully-connected graphs
            out_idx = torch.cat([out_idx, edge_index_full], dim=1)
            out_val = torch.cat([out_val, edge_attr_pad], dim=0)
            out_idx, out_val = torch_sparse.coalesce(
                out_idx, out_val, batch.num_nodes, batch.num_nodes,
                op="add"
            )

        batch.edge_index, batch.edge_attr = out_idx, out_val

        return batch
