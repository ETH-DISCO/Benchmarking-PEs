import torch
from torch import nn
from .graphit.layers import DiffTransformerEncoderLayer
from torch_geometric.graphgym.register import *
#
@register_layer('GraphiT')
class GraphTransformer(nn.Module):
    def __init__(self, in_size, d_model, nb_heads,
                 dim_feedforward=2048, dropout=0.1, nb_layers=4,
                 lap_pos_enc=False, lap_pos_enc_dim=0):
        super(GraphTransformer, self).__init__()
        self.embedding = nn.Linear(in_features=in_size,
                                   out_features=d_model,
                                   bias=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nb_heads, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, nb_layers)

    def forward(self, batch):
        # We permute the batch and sequence following pytorch
        # Transformer convention
        x = batch.x
        masks = None
        # x = x.permute(1, 0, 2)
        output = self.embedding(x)
        output = self.encoder(output, src_key_padding_mask=masks)
        # output = output.permute(1, 0, 2)
        # we make sure to correctly take the masks into account when pooling
        batch.x = output
        return batch

# :TODO Add DiffTrans
