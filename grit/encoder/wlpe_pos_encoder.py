import torch
import numpy as np
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torch_ppr import page_rank, personalized_page_rank
from tqdm import tqdm
from torch_geometric.utils import  add_self_loops
@register_node_encoder('WLPE')
class WLPENodeEncoder(torch.nn.Module):
    """Personalized PageRank embeddings
    we use the torch-ppr package to cover this
    source:
    https://github.com/mberr/torch-ppr/tree/main

    Args:
        dim_emb: Size of final node embedding
    """

    def __init__(self, dim_emb, expand_x = True):
        super().__init__()
        dim_in = cfg.share.dim_in
        pecfg = cfg.posenc_WLPE
        max_freqs = pecfg.dh # Num. eigenvectors (frequencies)
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None

        self.expand_x = expand_x
        self.dim_emb = pecfg.dim_pe
        self.add_selfloops = pecfg.add_self_loops

        self.linear_encoder = nn.Linear(max_freqs, self.dim_emb)


        self.dim_in = dim_in
    def forward(self, batch):


        pos_enc = batch.pos_enc.to(batch.x.device)
        # pos_enc = torch.tensor(batch.pos_enc).to("cuda:0")
        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors)
        pos_enc[empty_mask] = 0.  # (Num nodes) x (Num Eigenvectors)

        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)

        pos_enc = self.linear_encoder(pos_enc)
        h = batch.x
        # if self.expand_x:
        #     h = self.linear_encoder(pos_enc)
        # else:
        #     h = batch.x

        batch.x = torch.cat((h, pos_enc), 1)



        if self.add_selfloops:
            batch.edge_index, batch.edge_attr = add_self_loops(batch.edge_index, batch.edge_attr, num_nodes=batch.num_nodes, fill_value=0.)
        # Keep PE separate in a variable
        batch.pe_EquivStableLapPE = pos_enc
        return batch





