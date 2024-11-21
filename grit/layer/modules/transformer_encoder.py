import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerNodeEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.max_input_len = args.gt.max_input_len
        self.d_model = args.gt.d_model
        self.num_layer = args.gt.num_encoder_layers
        # Creating Transformer Encoder Model
        encoder_layer = nn.TransformerEncoderLayer(
            args.gt.d_model, args.gt.nhead, args.gt.dim_feedforward, args.gt.transformer_dropout, args.gt.transformer_activation
        )
        encoder_norm = nn.LayerNorm(args.gt.d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer, args.gt.num_encoder_layers, encoder_norm)

        self.norm_input = None
        if args.gt.transformer_norm_input:
            self.norm_input = nn.LayerNorm(args.gt.d_model)
        self.cls_embedding = None
        if args.gt.graph_pooling == "cls":
            self.cls_embedding = nn.Parameter(torch.randn([1, 1, args.gt.d_model], requires_grad=True))

    def forward(self, padded_h_node, src_padding_mask):
        """
        padded_h_node: n_b x B x h_d
        src_key_padding_mask: B x n_b
        """

        # (S, B, h_d), (B, S)

        if self.cls_embedding is not None:
            expand_cls_embedding = self.cls_embedding.expand(1, padded_h_node.size(1), -1)
            padded_h_node = torch.cat([padded_h_node, expand_cls_embedding], dim=0)

            zeros = src_padding_mask.data.new(src_padding_mask.size(0), 1).fill_(0)
            src_padding_mask = torch.cat([src_padding_mask, zeros], dim=1)
        if self.norm_input is not None:
            padded_h_node = self.norm_input(padded_h_node)

        transformer_out = self.transformer(padded_h_node, src_key_padding_mask=src_padding_mask)  # (S, B, h_d)

        return transformer_out, src_padding_mask