# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch_geometric.nn as gnn
from .SAT.layers import TransformerEncoderLayer
from einops import repeat
from torch_geometric.graphgym.register import *
import torch_geometric.utils as utils
from multiprocessing import Pool
class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, edge_index, complete_edge_index,
                subgraph_node_index=None, subgraph_edge_index=None,
                subgraph_edge_attr=None, subgraph_indicator_index=None, edge_attr=None, degree=None,
                ptr=None, return_attn=False):
        output = x

        for mod in self.layers:
            output = mod(output, edge_index, complete_edge_index,
                         edge_attr=edge_attr, degree=degree,
                         subgraph_node_index=subgraph_node_index,
                         subgraph_edge_index=subgraph_edge_index,
                         subgraph_indicator_index=subgraph_indicator_index,
                         subgraph_edge_attr=subgraph_edge_attr,
                         ptr=ptr,
                         return_attn=return_attn
                         )
        if self.norm is not None:
            output = self.norm(output)
        return output

@register_layer('SAT')
class SAT(nn.Module):
    def __init__(self, in_size, d_model, num_heads=8,
                 dim_feedforward=64, dropout=0.0, num_layers=4,
                 batch_norm=False, abs_pe=False, abs_pe_dim=0,
                 gnn_type="graph", se="gnn", use_edge_attr=False, num_edge_features=4,
                 in_embed=False, edge_embed=False, use_global_pool=True, max_seq_len=None,
                 global_pool='mean', **kwargs):
        super().__init__()

        self.abs_pe = abs_pe
        self.abs_pe_dim = abs_pe_dim
        if abs_pe and abs_pe_dim > 0:
            self.embedding_abs_pe = nn.Linear(abs_pe_dim, d_model)
        if in_embed:
            if isinstance(in_size, int):
                self.embedding = nn.Embedding(in_size, d_model)
            elif isinstance(in_size, nn.Module):
                self.embedding = in_size
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(in_features=in_size,
                                       out_features=d_model,
                                       bias=False)

        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 32)
            if edge_embed:
                if isinstance(num_edge_features, int):
                    self.embedding_edge = nn.Embedding(num_edge_features, edge_dim)
                else:
                    raise ValueError("Not implemented!")
            else:
                self.embedding_edge = nn.Linear(in_features=num_edge_features,
                                                out_features=edge_dim, bias=False)
        else:
            kwargs['edge_dim'] = None

        self.gnn_type = gnn_type
        self.se = se
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, se=se, **kwargs)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)

    def compute_degree(self, dataset):
        return 1. / torch.sqrt(1. + utils.degree(dataset.edge_index[0], dataset.num_nodes))


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        degree = self.compute_degree(data)
        node_indices = []
        edge_indices = []
        edge_attributes = []
        indicators = []
        edge_index_start = 0


        ### bottleneck
        for node_idx in range(data.num_nodes):
            sub_nodes, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(
                node_idx,
                2,
                data.edge_index,
                relabel_nodes=True,
                num_nodes=data.num_nodes
            )
            node_indices.append(sub_nodes)
            edge_indices.append(sub_edge_index + edge_index_start)
            indicators.append(torch.zeros(sub_nodes.shape[0]).fill_(node_idx))
            # if self.use_subgraph_edge_attr and graph.edge_attr is not None:
            edge_attributes.append(data.edge_attr[edge_mask])  # CHECK THIS DIDN"T BREAK ANYTHING
            edge_index_start += len(sub_nodes)



        subgraph_node_idx = torch.cat(node_indices)
        subgraph_edge_index = torch.cat(edge_indices, dim=1)
        subgraph_indicator = torch.cat(indicators)
        subgraph_edge_attr = torch.cat(edge_attributes)
        # num_subgraph_nodes = len(subgraph_edge_index)


        # node_depth = data.node_depth if hasattr(data, "node_depth") else None

        if self.se == "khopgnn":
            subgraph_node_index = subgraph_node_idx
            subgraph_edge_index = subgraph_edge_index
            subgraph_indicator_index = subgraph_indicator
            subgraph_edge_attr = subgraph_edge_attr
        else:
            subgraph_node_index = None
            subgraph_edge_index = None
            subgraph_indicator_index = None
            subgraph_edge_attr = None

        complete_edge_index = None
        abs_pe = data.abs_pe if hasattr(data, 'abs_pe') else None
        degree = degree
        output = self.embedding(x.to(device = x.device))

        # \
        #     if node_depth is None else self.embedding(x.to(dtype=torch.long, device = x.device), node_depth.view(-1, ))
        #
        #
        #

        if self.abs_pe and abs_pe is not None:
            abs_pe = self.embedding_abs_pe(abs_pe)
            output = output + abs_pe
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.embedding_edge(edge_attr)
            if subgraph_edge_attr is not None:
                subgraph_edge_attr = self.embedding_edge(subgraph_edge_attr)
        else:
            edge_attr = None
            subgraph_edge_attr = None
        output = self.encoder(
            output,
            edge_index,
            complete_edge_index,
            edge_attr=edge_attr,
            degree=degree,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index,
            subgraph_edge_attr=subgraph_edge_attr,
            ptr=data.ptr,
            return_attn=False
        )
        # readout step
        data.x = output
        return data