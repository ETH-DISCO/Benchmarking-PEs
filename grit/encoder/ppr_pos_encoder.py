import torch
import numpy as np
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torch_ppr import page_rank, personalized_page_rank
from tqdm import tqdm
import time
from torch_geometric.utils import  add_self_loops
@register_node_encoder('PPR')
class PPRNodeEncoder(torch.nn.Module):
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
        pecfg = cfg.posenc_PPR
        max_freqs = pecfg.eigen.max_freqs  # Num. eigenvectors (frequencies)
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None

        self.expand_x = expand_x
        self.dim_emb = dim_emb
        if expand_x:
            self.linear_encoder = nn.Linear(max_freqs, dim_emb - dim_in)

        self.add_selfloops = pecfg.add_self_loops

    def forward(self, batch):
        # n = batch.num_nodes
        # ppr_matrix = np.zeros((n, n))
        # ppr_matrix = personalized_page_rank(edge_index=batch.edge_index, indices=np.arange(n))
        # U, Sigma, VT = np.linalg.svd(ppr_matrix.cpu(), full_matrices=False)
        #
        # U_reduced = U[:, :cfg.posenc_PPR.eigen.max_freqs]
        tensors = [torch.tensor(array, device=batch.x.device) for array in batch.pos_enc]
        pos_enc = torch.cat(tensors, dim=0)
        # pos_enc = torch.tensor(batch.pos_enc).to("cuda:0")

        # empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors)
        # pos_enc[empty_mask] = 0.  # (Num nodes) x (Num Eigenvectors)

        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)

        if self.expand_x:
            h = self.linear_encoder(pos_enc)
        else:
            h = batch.x


        batch.x = torch.cat((h, pos_enc), 1)
        if self.add_selfloops:
            batch.edge_index, batch.edge_attr = add_self_loops(batch.edge_index, batch.edge_attr, num_nodes=batch.num_nodes, fill_value=0.)
        # Keep PE separate in a variable
        batch.pos_enc = pos_enc
        return batch





