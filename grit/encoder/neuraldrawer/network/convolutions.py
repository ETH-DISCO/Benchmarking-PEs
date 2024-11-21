import torch

from torch_geometric.nn import MessagePassing


class GRUEdgeConv(MessagePassing):
    def __init__(self, emb_dim, mlp_edge, aggr):
        super(GRUEdgeConv, self).__init__(aggr=aggr)
        self.rnn = torch.nn.GRUCell(emb_dim, emb_dim)
        self.mlp_edge = mlp_edge

    def forward(self, x, edge_index):
        out = self.rnn(self.propagate(edge_index, x=x), x)
        return out

    def message(self, x_j, x_i):
        concatted = torch.cat((x_j, x_i), dim=1)
        return self.mlp_edge(concatted)      

class GINEdgeConv(MessagePassing):
    def __init__(self, mlp, mlp_edge, aggr):
        super(GINEdgeConv, self).__init__(aggr=aggr)
        self.mlp = mlp
        self.mlp_edge = mlp_edge
        self.eps = torch.nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x, edge_index):
        #edge_index , _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x))
        return out
    
    def message(self, x_j, x_i):
        concatted = torch.cat((x_j, x_i), dim=1)
        return self.mlp_edge(concatted)

