import torch
import torch_geometric as pyg

import os.path
import re
import random

from tqdm.auto import tqdm
import numpy as np
import networkx as nx
import torch
import torch_geometric as pyg
from torch_geometric.utils import to_undirected, to_networkx
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.datasets import ZINC

import ssgetpy
import os
from scipy.io import mmread
import networkx as nx
from torch_geometric.utils import remove_self_loops, to_networkx, to_undirected, from_networkx
from torch_geometric.utils.convert import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_geometric.data import Data


from .transforms import convert_for_DeepGD, convert_for_stress, pmds_layout, filter_connected 

from graph_coarsening.coarsening_utils import *
import graph_coarsening.graph_utils
import pygsp as gsp
from scipy.spatial import Delaunay


DATA_ROOT = "data"
DEEPGD_DATA_ROOT = "deepgd_data"

def get_dataset(dataset_name, for_DeepGD=False):
    if for_DeepGD:
        data_root = DEEPGD_DATA_ROOT
        transform = convert_for_DeepGD
    else:
        data_root = DATA_ROOT
        transform = convert_for_stress
    
    if dataset_name == 'rome':
        random.seed(12345)
        np.random.seed(12345)
        if for_DeepGD:
            dataset = RomeDatasetDeepGD(layout_initializer=pmds_layout)
        else:
            dataset = RomeDataset()
        datalist = list(dataset)
        random.shuffle(datalist)
        train = datalist[:10000]
        val = datalist[11000:12000]
        test = datalist[10000:11000]
    elif dataset_name == 'cluster':
        train = list(GNNBenchmarkDataset(root=data_root+'/CLUSTER', name='CLUSTER', split='train', pre_transform=transform))
        val = list(GNNBenchmarkDataset(root=data_root+'/CLUSTER', name='CLUSTER', split='val', pre_transform=transform))
        test = list(GNNBenchmarkDataset(root=data_root+'/CLUSTER', name='CLUSTER', split='test', pre_transform=transform))
    elif dataset_name == 'pattern':
        train = list(GNNBenchmarkDataset(root=data_root+'/PATTERN', name='PATTERN', split='train', pre_transform=transform))
        val = list(GNNBenchmarkDataset(root=data_root+'/PATTERN', name='PATTERN', split='val', pre_transform=transform))
        test = list(GNNBenchmarkDataset(root=data_root+'/PATTERN', name='PATTERN', split='test', pre_transform=transform))
    elif dataset_name == 'zinc':
        train = list(ZINC(root=data_root+'/ZINC', subset=True, split='train', pre_transform=transform))
        val = list(ZINC(root=data_root+'/ZINC', subset=True, split='val', pre_transform=transform))
        test = list(ZINC(root=data_root+'/ZINC', subset=True, split='test', pre_transform=transform))
    elif dataset_name == 'mnist':
        train = list(GNNBenchmarkDataset(root=data_root+'/MNIST', name='MNIST', split='train', pre_transform=transform))
        val = list(GNNBenchmarkDataset(root=data_root+'/MNIST', name='MNIST', split='val', pre_transform=transform))
        test = list(GNNBenchmarkDataset(root=data_root+'/MNIST', name='MNIST', split='test', pre_transform=transform))
    elif dataset_name == 'cifar10':
        train = list(GNNBenchmarkDataset(root=data_root+'/CIFAR10', name='CIFAR10', split='train', pre_transform=transform))
        val = list(GNNBenchmarkDataset(root=data_root+'/CIFAR10', name='CIFAR10', split='val', pre_transform=transform))
        test = list(GNNBenchmarkDataset(root=data_root+'/CIFAR10', name='CIFAR10', split='test', pre_transform=transform))
    elif dataset_name == 'mixed':
        rome_train, rome_val, rome_test = get_dataset('rome')
        cluster_train, cluster_val, cluster_test = get_dataset('cluster')
        pattern_train, pattern_val, pattern_test = get_dataset('pattern')
        zinc_train, zinc_val, zinc_test = get_dataset('zinc')
        mnist_train, mnist_val, mnist_test = get_dataset('mnist')
        cifar10_train, cifar10_val, cifar10_test = get_dataset('cifar10')
        train_datasets = [rome_train, cluster_train, pattern_train, zinc_train, mnist_train, cifar10_train]
        val_datasets = [rome_val, cluster_val, pattern_val, zinc_val, mnist_val, cifar10_val]
        test_datasets = [rome_test, cluster_test, pattern_test, zinc_test, mnist_test, cifar10_test]
        train, val, test = [], [], []
        for train_set in train_datasets:
            train += train_set[:10000]
        for val_set in val_datasets:
            val += val_set[:1000]
        for test_set in test_datasets:
            test += test_set[:1000]
    elif dataset_name == 'suitesparse':
        destpath=f'{data_root}/suitsparse/'
        ssgetpy.search(limit=5000, rowbounds=(100,1000), colbounds=(100,1000)).download(extract=True, destpath=destpath)
        mtx_folders = os.listdir(destpath)
        mtx_folders.sort()
        datalist = []
        for folder in mtx_folders:
            graph_file = [graph_file for graph_file in os.listdir(destpath + '/' + folder) if graph_file.endswith('.mtx')][0]
            a = mmread(destpath + '/' + folder + '/' + graph_file)
            if a.shape[0] == a.shape[1]:
                graph = nx.Graph(a)
                if nx.is_connected(graph) and graph.number_of_nodes() >= 100 and graph.number_of_nodes() < 1000 and graph.number_of_edges() > 0:
                    pyg_graph = from_networkx(graph)
                    pyg_graph.edge_index,_ = remove_self_loops(edge_index=pyg_graph.edge_index)
                    pyg_graph.edge_index = to_undirected(edge_index=pyg_graph.edge_index)
                    datalist.append(pyg_graph)

        datalist = [transform(data) for data in datalist]
        random.Random(42).shuffle(datalist)
        
        train = datalist[0:int(len(datalist)*0.8)]
        val = datalist[int(len(datalist)*0.8):int(len(datalist)*0.9)]
        test = datalist[int(len(datalist)*0.9):]

    elif dataset_name == 'delaunay':
        random.seed(0)
        np.random.seed(0)
        datalist = []
        for i in range(240):
            graph_size = random.randint(100, 1000)
            datalist.append(random_delaunay_graph(graph_size))
        datalist = [transform(data) for data in datalist]
        train = datalist[:200]
        val = datalist[200:220]
        test = datalist[220:]
    
    elif dataset_name == 'delaunay_deepgd': # Used to train DeepGD as the standard delaunay dataset is too big for training
        random.seed(0)
        np.random.seed(0)
        datalist = []
        for i in range(240):
            graph_size = random.randint(10, 200)
            datalist.append(random_delaunay_graph(graph_size))
        datalist = [transform(data) for data in datalist]
        train = datalist[:200]
        val = datalist[200:220]
        test = datalist[220:]
    else:
        print('Unrecognized dataset: ' + dataset_name)
        exit(1)

    return train, val, test


def random_delaunay_graph(num_nodes):
    points = np.random.rand(num_nodes, 2)

    triangulation = Delaunay(points).vertex_neighbor_vertices
    G = nx.Graph()

    for i in range(num_nodes):
        G.add_node(i)

    for i in range(len(triangulation[0])-1):
        for j in range(triangulation[0][i], triangulation[0][i+1]):
            G.add_edge(i, triangulation[1][j])

    data = from_networkx(G)
    data.x = torch.zeros(num_nodes)

    return data


def get_mtx_graphs(mtx_folders, destpath):
    datalist = []
    for folder in mtx_folders:
        graph_file = [graph_file for graph_file in os.listdir(destpath + '/' + folder) if graph_file.endswith('.mtx')][0]
        a = mmread(destpath + '/' + folder + '/' + graph_file)
        if a.shape[0] == a.shape[1]:
            graph = nx.Graph(a)
            if nx.is_connected(graph) and graph.number_of_nodes() < 1000 and graph.number_of_nodes() > 100 and graph.number_of_edges() > 0:
                print(graph.number_of_nodes())
                pyg_graph = from_networkx(graph)
                pyg_graph.edge_index,_ = remove_self_loops(edge_index=pyg_graph.edge_index)
                pyg_graph.edge_index = to_undirected(edge_index=pyg_graph.edge_index)
                datalist.append(pyg_graph)

    return datalist


class RomeDataset(pyg.data.InMemoryDataset):
    def __init__(self, *,
                 url='http://www.graphdrawing.org/download/rome-graphml.tgz',
                 root=f'{DATA_ROOT}/Rome',
                 layout_initializer=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.url = url
        self.initializer = layout_initializer or nx.drawing.random_layout
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        metafile = "rome/Graph.log"
        if os.path.exists(metadata_path := f'{self.raw_dir}/{metafile}'):
            return list(map(lambda f: f'rome/{f}.graphml',
                            self.get_graph_names(metadata_path)))
        else:
            return [metafile]

    @property
    def processed_file_names(self):
        return ['data.pt']

    @classmethod
    def get_graph_names(cls, logfile):
        with open(logfile) as fin:
            for line in fin.readlines():
                if match := re.search(r'name: (grafo\d+\.\d+)', line):
                    yield f'{match.group(1)}'

    def process_raw(self):
        graphmls = sorted(self.raw_paths,
                          key=lambda x: int(re.search(r'grafo(\d+)', x).group(1)))
        for file in tqdm(graphmls, desc=f"Loading graphs"):
            G = nx.read_graphml(file)
            if nx.is_connected(G):
                yield nx.convert_node_labels_to_integers(G)

    def convert(self, G):
        apsp = dict(nx.all_pairs_shortest_path_length(G))
        init_pos = torch.tensor(np.array(list(self.initializer(G).values())))
        full_edges, attr_d = zip(*[((u, v), d) for u in apsp for v, d in apsp[u].items()])
        raw_edge_index = pyg.utils.to_undirected(torch.tensor(list(G.edges)).T)
        full_edge_index, d = pyg.utils.remove_self_loops(*pyg.utils.to_undirected(
            torch.tensor(full_edges).T, torch.tensor(attr_d)
        ))
        start,end = [i for i,j in G.edges],[j for i,j in G.edges]
        edges_actual = torch.tensor([start,end])
        edges_actual = to_undirected(edges_actual)
        k = 1 / d ** 2
        full_edge_attr = torch.stack([d, k], dim=-1)
        return pyg.data.Data(
            x=init_pos,
            edge_index=edges_actual,
            full_edge_index=full_edge_index,
            full_edge_attr=full_edge_attr,
        )

    def download(self):
        pyg.data.download_url(self.url, self.raw_dir)
        pyg.data.extract_tar(f'{self.raw_dir}/rome-graphml.tgz', self.raw_dir)

    def process(self):
        data_list = map(self.convert, self.process_raw())

        if self.pre_filter is not None:
            data_list = filter(self.pre_filter, data_list)

        if self.pre_transform is not None:
            data_list = map(self.pre_transform, data_list)

        data, slices = self.collate(list(data_list))
        torch.save((data, slices), self.processed_paths[0])
        

def get_edges(node_pos, batch):
    edges = node_pos[batch.edge_index.T]
    return edges[:, 0, :], edges[:, 1, :]


def get_full_edges(node_pos, batch):
    edges = node_pos[batch.full_edge_index.T]
    return edges[:, 0, :], edges[:, 1, :]

class RomeDatasetDeepGD(pyg.data.InMemoryDataset):
    def __init__(self, *,
                 url='http://www.graphdrawing.org/download/rome-graphml.tgz',
                 root=f'{DEEPGD_DATA_ROOT}/Rome',
                 layout_initializer=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.url = url
        self.initializer = layout_initializer or nx.drawing.random_layout
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        metafile = "rome/Graph.log"
        if os.path.exists(metadata_path := f'{self.raw_dir}/{metafile}'):
            return list(map(lambda f: f'rome/{f}.graphml',
                            self.get_graph_names(metadata_path)))
        else:
            return [metafile]

    @property
    def processed_file_names(self):
        return ['data.pt']

    @classmethod
    def get_graph_names(cls, logfile):
        with open(logfile) as fin:
            for line in fin.readlines():
                if match := re.search(r'name: (grafo\d+\.\d+)', line):
                    yield f'{match.group(1)}'

    def process_raw(self):
        graphmls = sorted(self.raw_paths,
                          key=lambda x: int(re.search(r'grafo(\d+)', x).group(1)))
        for file in tqdm(graphmls, desc=f"Loading graphs"):
            G = nx.read_graphml(file)
            if nx.is_connected(G):
                yield nx.convert_node_labels_to_integers(G)

    def convert(self, G):
        apsp = dict(nx.all_pairs_shortest_path_length(G))
        init_pos = torch.tensor(np.array(list(self.initializer(G).values())), dtype=torch.float)
        full_edges, attr_d = zip(*[((u, v), d) for u in apsp for v, d in apsp[u].items()])
        raw_edge_index = pyg.utils.to_undirected(torch.tensor(list(G.edges)).T)
        full_edge_index, d = pyg.utils.remove_self_loops(*pyg.utils.to_undirected(
            torch.tensor(full_edges).T, torch.tensor(attr_d)
        ))
        k = 1 / d ** 2
        full_edge_attr = torch.stack([d, k], dim=-1)
        return pyg.data.Data(
            G=G,
            x=init_pos,
            init_pos=init_pos,
            edge_index=full_edge_index,
            edge_attr=full_edge_attr,
            raw_edge_index=raw_edge_index,
            full_edge_index=full_edge_index,
            full_edge_attr=full_edge_attr,
            d=d,
            n=G.number_of_nodes(),
            m=G.number_of_edges(),
        )

    def download(self):
        pyg.data.download_url(self.url, self.raw_dir)
        pyg.data.extract_tar(f'{self.raw_dir}/rome-graphml.tgz', self.raw_dir)

    def process(self):
        data_list = map(self.convert, self.process_raw())

        if self.pre_filter is not None:
            data_list = filter(self.pre_filter, data_list)

        if self.pre_transform is not None:
            data_list = map(self.pre_transform, data_list)

        data, slices = self.collate(list(data_list))
        torch.save((data, slices), self.processed_paths[0])
