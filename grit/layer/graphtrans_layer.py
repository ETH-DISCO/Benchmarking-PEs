import torch
import torch.nn as nn
from loguru import logger
from .modules.masked_transformer_encoder import MaskedOnlyTransformerEncoder
from .modules.transformer_encoder import TransformerNodeEncoder
from .modules.utils import pad_batch, unpad_batch
from .modules.base_model import BaseModel
from torch_geometric.graphgym.register import *
from torch_geometric.utils import to_dense_adj
@register_layer('GraphTrans')
class TransformerGNN(BaseModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input2transformer = nn.Linear(args.gt.graph_input_dim,
                                           args.gt.d_model) if args.gt.graph_input_dim is not None else None
        self.transformer_encoder = TransformerNodeEncoder(args)
        self.masked_transformer_encoder = MaskedOnlyTransformerEncoder(args)
        self.transformer2gnn = nn.Linear(args.gt.d_model, args.gt.gnn_emb_dim)
        self.num_encoder_layers = args.gt.num_encoder_layers
        self.num_encoder_layers_masked = args.gt.num_encoder_layers_masked


    def forward(self, batched_data, perturb=None):
        x = batched_data.x

        tmp = self.input2transformer(x)
        padded_h_node, src_padding_mask, num_nodes, mask, max_num_nodes = pad_batch(
            x, batched_data.batch, self.args.gt.max_input_len, get_mask=True
        )  # Pad in the front

        transformer_out = padded_h_node

        batch_size = batched_data.num_graphs
        max_num_nodes = max(
            batched_data.batch.bincount().tolist())  # The maximum number of nodes in any graph in the batch

        if self.num_encoder_layers > 0:
            transformer_out, _ = self.transformer_encoder(transformer_out, src_padding_mask)  # [s, b, h], [b, s]

        h_node = unpad_batch(transformer_out, tmp, num_nodes, mask, max_num_nodes)
        batched_data.x = self.transformer2gnn(h_node)
        return batched_data
