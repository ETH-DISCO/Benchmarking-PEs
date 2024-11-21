from tqdm import tqdm
import torch
from torch_geometric.transforms.add_positional_encoding import AddLaplacianEigenvectorPE
import math
import random
from torch_geometric.nn import MessagePassing
import torch_scatter
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
    to_undirected,
    is_undirected,
)
from torch_geometric.data import Data
from typing import Any, Optional
import numpy as np
#import cupy as cp
try:
    import cupy as cp
except ImportError:
    pass

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
#torch.set_default_device(device)


class BFSConv(MessagePassing):
    def __init__(self, aggr = "min"):
        super().__init__(aggr=aggr)

    def forward(self, distances, edge_index):
        msg = self.propagate(edge_index, x=distances)
        return torch.minimum(msg, distances)

    def message(self, x_j):
        return x_j + 1

class BFS(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = BFSConv()
    
    def forward(self, data, distances, max_iterations):
        edge_index = data.edge_index

        iteration = 0
        while float('Inf') in distances and iteration < max_iterations:
            distances = self.conv(distances, edge_index)
            iteration += 1
        
        if iteration == max_iterations:
            print('Warning: Check if the graph is connected!')

        return distances

def add_node_attr(data: Data, value: Any,
                  attr_name: Optional[str] = None) -> Data:
    # TODO Move to `BaseTransform`.
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data

class AddLaplacian(BaseTransform):
    r"""Adds the Laplacian eigenvector positional encoding from the
    `"Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
    paper to the given graph
    (functional name: :obj:`add_laplacian_eigenvector_pe`).

    Args:
        k (int): The number of non-trivial eigenvectors to consider.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"laplacian_eigenvector_pe"`)
        is_undirected (bool, optional): If set to :obj:`True`, this transform
            expects undirected graphs as input, and can hence speed up the
            computation of eigenvectors. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :meth:`scipy.sparse.linalg.eigs` (when :attr:`is_undirected` is
            :obj:`False`) or :meth:`scipy.sparse.linalg.eigsh` (when
            :attr:`is_undirected` is :obj:`True`).
    """
    def __init__(
        self,
        k: int,
        attr_name: Optional[str] = 'laplacian_eigenvector_pe',
        use_cupy=False,
        **kwargs,
    ):
        self.k = k
        self.attr_name = attr_name
        self.kwargs = kwargs
        self.use_cupy = use_cupy

    def __call__(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        edge_index, edge_weight = get_laplacian(
            data.edge_index,
            data.edge_weight,
            normalization='sym',
            num_nodes=num_nodes,
        )
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
        L_np = L.toarray()
        
        #L_cp = cp.array(L_np)
        #eig_vals, eig_vecs = cp.linalg.eigh(L_cp)
        #eig_vecs = cp.real(eig_vecs[:, eig_vals.argsort()])
        #pe = torch.from_numpy(cp.asnumpy(eig_vecs[:, 1:self.k + 1])).to(device)

        if device == 'cpu' or not self.use_cupy:
            eig_vals, eig_vecs = np.linalg.eigh(L_np)
            eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
            pe = torch.from_numpy(eig_vecs[:, 1:self.k + 1])

        else:
            L_cp = cp.array(L_np)
            eig_vals, eig_vecs = cp.linalg.eigh(L_cp)
            eig_vecs = cp.real(eig_vecs[:, eig_vals.argsort()])
            pe = torch.from_numpy(cp.asnumpy(eig_vecs[:, 1:self.k + 1])).to(device)


        #eig_vals,eig_vecs = np.linalg.eigh(L_np)

        #eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
        #pe = torch.from_numpy(eig_vecs[:, 1:self.k + 1])
        sign = -1 + 2 * torch.randint(0, 2, (self.k, ))
        sign = sign.to(pe.device)
        pe *= sign

        data = add_node_attr(data, pe.to(data.x.device), attr_name=self.attr_name)
        return data


def compute_positional_encodings(dataset, num_beacons, encoding_size_per_beacon):
    bfs = BFS()
    for graph in dataset:
        starting_nodes = random.sample(range(graph.num_nodes), num_beacons)
        distances = torch.empty(graph.num_nodes, num_beacons, device = graph.x.device).fill_(float('Inf'))
        for i in range(num_beacons):
            distances[starting_nodes[i], i] = 0
        distance_encodings = torch.zeros((graph.num_nodes, num_beacons * encoding_size_per_beacon), dtype=torch.float)
        bfs_distances = bfs(graph, distances, graph.num_nodes)
    
        div_term = torch.exp(torch.arange(0, encoding_size_per_beacon, 2) * (-math.log(10000.0) / encoding_size_per_beacon)).to(bfs_distances.device)
        pes = []
        for beacon_index in range(num_beacons):
            pe = torch.zeros(graph.num_nodes, encoding_size_per_beacon, device=bfs_distances.device)
            pe[:, 0::2] = torch.sin(bfs_distances[:, beacon_index].unsqueeze(1) * div_term)
            pe[:, 1::2] = torch.cos(bfs_distances[:, beacon_index].unsqueeze(1) * div_term)
            pes.append(pe)
        graph.pe = torch.cat(pes,1)
    
    return dataset

def compute_positional_encodings_batch(batch, num_beacons, encoding_size_per_beacon):
    bfs = BFS()
    graph_sizes = torch_scatter.scatter(torch.ones(batch.batch.shape[0]), batch.batch.cpu()).tolist()
    starting_nodes_per_graph = [random.sample(range(int(num_nodes)), num_beacons) for num_nodes in graph_sizes]
    
    graph_size_acc = 0

    distances = torch.empty(batch.x.shape[0], num_beacons, device = batch.x.device).fill_(float('Inf'))
    for i in range(0, len(graph_sizes)):
        for j in range(num_beacons):
            distances[starting_nodes_per_graph[i][j] + int(graph_size_acc), j] = 0
        graph_size_acc += graph_sizes[i]

    distance_encodings = torch.zeros((batch.x.shape[0], num_beacons * encoding_size_per_beacon), dtype=torch.float)
    bfs_distances = bfs(batch, distances, max(graph_sizes))
    
    div_term = torch.exp(torch.arange(0, encoding_size_per_beacon, 2) * (-math.log(10000.0) / encoding_size_per_beacon)).to(bfs_distances.device)
    pes = []
    for beacon_index in range(num_beacons):
        pe = torch.zeros(batch.x.shape[0], encoding_size_per_beacon, device=bfs_distances.device)
        pe[:, 0::2] = torch.sin(bfs_distances[:, beacon_index].unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(bfs_distances[:, beacon_index].unsqueeze(1) * div_term)
        pes.append(pe)
    pes_tensor = torch.cat(pes,1)
    
    return pes_tensor


def preprocess_dataset(datalist, config):
    spectrals = []
    if config.use_beacons:
        datalist = compute_positional_encodings(datalist, config.num_beacons, config.encoding_size_per_beacon)
    for idx in range(len(datalist)):
        eigenvecs = config.laplace_eigvec
        beacons = torch.zeros(datalist[idx].num_nodes, 0, dtype=torch.float, device=datalist[idx].x.device)
        if config.use_beacons:
            beacons = datalist[idx].pe
        spectral_features = torch.zeros(datalist[idx].num_nodes, 0, dtype=torch.float, device=datalist[idx].x.device)
        if eigenvecs > 0:
            pe_transform = AddLaplacian(k=eigenvecs, attr_name="laplace_ev", is_undirected=True, use_cupy=config.use_cupy)
            datalist[idx] = pe_transform(datalist[idx])
            spectral_features = datalist[idx].laplace_ev
        dim = datalist[idx].x.size(dim=0)
        x = torch.rand(dim, config.random_in_channels, dtype=torch.float, device=datalist[idx].x.device)
        datalist[idx].x = torch.cat((x, beacons, spectral_features), dim=1)
        datalist[idx].x_orig = torch.clone(datalist[idx].x)
    return datalist

def reset_randomized_features_batch(batch, config):
    rand_features = torch.rand(batch.x.shape[0], config.random_in_channels, dtype=torch.float, device=batch.x.device)
    batch.x[:,:config.random_in_channels] = rand_features
    if config.use_beacons:
        pes = compute_positional_encodings_batch(batch, config.num_beacons, config.encoding_size_per_beacon)
        batch.x[:,config.random_in_channels:config.random_in_channels+pes.size(dim=1)] = pes
        batch.pe = pes
    batch.x_orig = torch.clone(batch.x)

    return batch


def reset_eigvecs(datalist, config):
    pe_transform = AddLaplacian(k=config.laplace_eigvec, attr_name="laplace_ev", is_undirected=True, use_cupy=config.use_cupy)
    for idx in range(len(datalist)):
        datalist[idx] = pe_transform(datalist[idx])
        spectral_features = datalist[idx].laplace_ev
        datalist[idx].x[:,-config.laplace_eigvec:] = spectral_features
        datalist[idx].x_orig[:,-config.laplace_eigvec:] = spectral_features
    return datalist

