import torch
from torch import nn
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.graphgym.register import *
#
@register_layer('UniMP')
class UniMP(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads,
                 concat, dropout=0.1, beta=True):
        super(UniMP, self).__init__()
        self.layer = TransformerConv(in_channels=in_dim,
                                     out_channels=out_dim,
                                     heads=num_heads,
                                     concat=concat, # Bool, default True
                                     dropout=dropout, # default 0.6
                                     beta=beta) # Boolean, default True
    def forward(self, batch):
        x = self.layer(batch.x, batch.edge_index, batch.edge_attr)
        batch.x = x
        return batch
