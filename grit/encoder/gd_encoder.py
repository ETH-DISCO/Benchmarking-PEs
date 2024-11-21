import torch
import torch.nn as nn
from attrdict import AttrDict
import torch_geometric
import json
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
from .neuraldrawer.network.preprocessing import preprocess_dataset
from .neuraldrawer.network.model import get_model

import os

@register_node_encoder('GDNodeEncoder')
class GDNodeEncoder(torch.nn.Module):
    """Graph Drawing node encoder.

    Embeddings of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with embeddings.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        dim_in = cfg.share.dim_in  # Expected original input node features dim

        pecfg = cfg.posenc_GD
        model_path = pecfg.model_path
        config_path = pecfg.config_path
        model_type = pecfg.model.lower()  # Encoder NN model type for PEs
        self.measurement = pecfg.get("measurement", False)
        dim_pe = pecfg.dim_pe  # Size of embedding
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable

        if dim_emb - dim_pe < 1:
            raise ValueError(f"LapPE size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x


        with open(os.getcwd() + config_path, 'r') as f:
            self.config = AttrDict(json.load(f))
        self.eval_model = get_model(self.config)
        self.eval_model.load_state_dict(torch.load(os.getcwd() + model_path))
        for param in self.eval_model.parameters():
            param.requires_grad = False
        self.eval_model.eval()

        n_layers = pecfg.layers
        if pecfg.use_embeddings:
            in_dim = self.config.hidden_dimension
            self.use_embedding = True
        else:
            in_dim = self.config.out_dim
            self.use_embedding = False

        activation = nn.ReLU
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(in_dim, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(in_dim, 2 * dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pos_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pos_encoder = nn.Linear(in_dim, dim_pe)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, batch):
        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x

        if not self.measurement:
            with torch.no_grad():
                original_edges = batch.edge_index
                batch.x_orig = batch.x
                data_list = batch.to_data_list()
                for i in range(len(data_list)):
                    data_list[i].edge_index = torch_geometric.utils.to_undirected(data_list[i].edge_index)
                data_list = preprocess_dataset(data_list, self.config)
                batch = torch_geometric.data.Batch.from_data_list(data_list).to(h.device)

                pred, pos_enc = self.eval_model(batch, 20, return_layers=True)
                batch.edge_index = original_edges

                if self.use_embedding:
                    pos_enc = pos_enc[-1].detach()
                else:
                    pos_enc = pred.detach()
        else:
            pos_enc = batch.pos_enc


        pos_enc = self.pos_encoder(pos_enc)

        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            batch.pe_GD = pos_enc
        return batch
