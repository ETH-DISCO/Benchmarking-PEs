import torch
from torch import nn
import torch_scatter

import numpy as np
import torch

class NormalizedStress(nn.Module):
    def __init__(self, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce

    def forward(self, node_pos, batch):
        start, end = get_full_edges(node_pos, batch)
        eu = (start - end).norm(dim=1)
        d = batch.full_edge_attr[:, 0]

        index = batch.batch[batch.full_edge_index[0]]
        scale = torch_scatter.scatter((eu/d)**2, index) / torch_scatter.scatter(eu/d, index)
        scaled_pos = node_pos / scale[batch.batch][:, None]

        start, end = get_full_edges(scaled_pos, batch)
        eu = (start - end).norm(dim=1)

        edge_stress = eu.sub(d).abs().div(d).square()
        index = batch.batch[batch.full_edge_index[0]]
        graph_stress = torch_scatter.scatter(edge_stress, index)
        graph_sizes = torch_scatter.scatter(torch.ones(batch.batch.shape[0], device=batch.batch.device), batch.batch)
        norm_factor = graph_sizes * graph_sizes
        graph_stress = torch.div(graph_stress, norm_factor)

        return graph_stress if self.reduce is None else self.reduce(graph_stress)

    
def get_full_edges(node_pos, batch):
    edges = node_pos[batch.full_edge_index.T]
    return edges[:, 0, :], edges[:, 1, :]


class ScaledStress(nn.Module):
    def __init__(self, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce
        
    def forward(self, node_pos, batch):
        start, end = get_full_edges(node_pos, batch)
        eu = (start - end).norm(dim=1)
        d = batch.full_edge_attr[:, 0]
        
        index = batch.batch[batch.full_edge_index[0]]
        scale = torch_scatter.scatter((eu/d)**2, index) / torch_scatter.scatter(eu/d, index)
        scaled_pos = node_pos / scale[batch.batch][:, None]

        start, end = get_full_edges(scaled_pos, batch)
        eu = (start - end).norm(dim=1)

        edge_stress = eu.sub(d).abs().div(d).square()
        index = batch.batch[batch.full_edge_index[0]]
        graph_stress = torch_scatter.scatter(edge_stress, index)
        return graph_stress if self.reduce is None else self.reduce(graph_stress)

class Stress(nn.Module):
    def __init__(self, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce
        
    def forward(self, node_pos, batch):
        start, end = get_full_edges(node_pos, batch)
        eu = (start - end).norm(dim=1)
        d = batch.full_edge_attr[:, 0]
        edge_stress = eu.sub(d).abs().div(d).square()
        index = batch.batch[batch.full_edge_index[0]]
        graph_stress = torch_scatter.scatter(edge_stress, index)
        return graph_stress if self.reduce is None else self.reduce(graph_stress)

