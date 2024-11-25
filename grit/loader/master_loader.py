import logging
import os.path as osp
import time
import warnings
import os
import zipfile
from functools import partial
from ..transform.expander_edges import generate_random_expander
from yacs.config import CfgNode as CN
import psutil

from grit.utils import get_device
from urllib.parse import urljoin
from typing import List, Optional
import gc
import requests
from tqdm import tqdm, trange
import numpy as np

import torch
import torch_geometric.transforms as T
from numpy.random import default_rng
from grit.head.identity import IdentityHead
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import (GNNBenchmarkDataset, Planetoid, TUDataset,
                                      WikipediaNetwork, ZINC)
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.model_builder import GraphGymModule
from torch_geometric.graphgym.loader import load_pyg, load_ogb, set_dataset_attr
from torch_geometric.graphgym.register import register_loader

from grit.transform.transforms import (VirtualNodePatchSingleton,
                                           clip_graphs_to_size,
                                           concat_x_and_pos,
                                           pre_transform_in_memory, typecast_x)
from grit.encoder.gnn_encoder import gpse_process_batch
from grit.loader.dataset.aqsol_molecules import AQSOL
from grit.loader.dataset.coco_superpixels import COCOSuperpixels
from grit.loader.dataset.malnet_tiny import MalNetTiny
from grit.loader.dataset.voc_superpixels import VOCSuperpixels
from grit.loader.split_generator import (prepare_splits,
                                         set_dataset_splits)
from grit.transform.posenc_stats import compute_posenc_stats, ComputePosencStat
from grit.transform.transforms import (pre_transform_in_memory,
                                       typecast_x, concat_x_and_pos,
                                       clip_graphs_to_size)

from grit.transform.dist_transforms import (add_dist_features, add_reverse_edges,
                                                 add_self_loops, effective_resistances,
                                                 effective_resistance_embedding,
                                                 effective_resistances_from_embedding)
from torch_geometric.data import DataLoader, Data
from torch_geometric.utils import degree
from torch_geometric.utils.convert import from_networkx
import networkx as nx

# Synthetic datasets

class SymmetrySet:
    def __init__(self):
        self.hidden_units = 0
        self.num_classes = 0
        self.num_features = 0
        self.num_nodes = 0

    def addports(self, data):
        data.ports = torch.zeros(data.num_edges, 1)
        degs = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)  # out degree of all nodes
        for n in range(data.num_nodes):
            deg = degs[n]
            ports = np.random.permutation(int(deg))
            for i, neighbor in enumerate(data.edge_index[1][data.edge_index[0] == n]):
                nb = int(neighbor)
                data.ports[torch.logical_and(data.edge_index[0] == n, data.edge_index[1] == nb), 0] = float(ports[i])
        return data

    def makefeatures(self, data):
        data.x = torch.ones((data.num_nodes, 1))
        data.id = torch.tensor(np.random.permutation(np.arange(data.num_nodes))).unsqueeze(1)
        return data

    def makedata(self):
        pass


class LimitsOne(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 4
        self.num_nodes = 8
        self.graph_class = False

    def makedata(self):
        n_nodes = 16  # There are two connected components, each with 8 nodes

        ports = [1, 1, 2, 2] * 8
        colors = [0, 1, 2, 3] * 4

        y = torch.tensor([0] * 8 + [1] * 8)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
                                    13, 14, 14, 15, 15, 8],
                                   [1, 0, 2, 1, 3, 2, 0, 3, 5, 4, 6, 5, 7, 6, 4, 7, 9, 8, 10, 9, 11, 10, 12, 11, 13, 12,
                                    14, 13, 15, 14, 8, 15]], dtype=torch.long)
        x = torch.zeros((n_nodes, 4))
        x[range(n_nodes), colors] = 1

        data = Data(x=x, edge_index=edge_index, y=y)
        data.id = torch.tensor(np.random.permutation(np.arange(n_nodes))).unsqueeze(1)
        data.ports = torch.tensor(ports).unsqueeze(1)
        data.train_mask = torch.ones(16, dtype=torch.bool)  # All nodes are in the train set
        data.test_mask = torch.ones(16, dtype=torch.bool)  # All nodes are also in the test set
        data.val_mask = torch.ones(16, dtype=torch.bool)
        return [data]


class LimitsTwo(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 4
        self.num_nodes = 8
        self.graph_class = False

    def makedata(self):
        n_nodes = 16  # There are two connected components, each with 8 nodes

        ports = ([1, 1, 2, 2, 1, 1, 2, 2] * 2 + [3, 3, 3, 3]) * 2
        colors = [0, 1, 2, 3] * 4
        y = torch.tensor([0] * 8 + [1] * 8)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 1, 3, 5, 7, 8, 9, 9, 10, 10, 11, 11,
                                    8, 12, 13, 13, 14, 14, 15, 15, 12, 9, 15, 11, 13],
                                   [1, 0, 2, 1, 3, 2, 0, 3, 5, 4, 6, 5, 7, 6, 4, 7, 3, 1, 7, 5, 9, 8, 10, 9, 11, 10, 8,
                                    11, 13, 12, 14, 13, 15, 14, 12, 15, 15, 9, 13, 11]], dtype=torch.long)
        x = torch.zeros((n_nodes, 4))
        x[range(n_nodes), colors] = 1

        data = Data(x=x, edge_index=edge_index, y=y)
        data.id = torch.tensor(np.random.permutation(np.arange(n_nodes))).unsqueeze(1)
        data.ports = torch.tensor(ports).unsqueeze(1)
        data.train_mask = torch.ones(16, dtype=torch.bool)  # All nodes are in the train set
        data.test_mask = torch.ones(16, dtype=torch.bool)  # All nodes are also in the test set
        data.val_mask = torch.ones(16, dtype=torch.bool)
        return [data]


class Triangles(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 1
        self.num_nodes = 60
        self.graph_class = False

    def makedata(self):
        size = self.num_nodes
        generated = False
        while not generated:
            nx_g = nx.random_degree_sequence_graph([3] * size)
            data = from_networkx(nx_g)
            labels = [0] * size
            for n in range(size):
                for nb1 in data.edge_index[1][data.edge_index[0] == n]:
                    for nb2 in data.edge_index[1][data.edge_index[0] == n]:
                        if torch.logical_and(data.edge_index[0] == nb1, data.edge_index[1] == nb2).any():
                            labels[n] = 1
            generated = labels.count(0) >= 20 and labels.count(1) >= 20
        data.y = torch.tensor(labels)

        data = self.addports(data)
        data = self.makefeatures(data)
        data.train_mask = torch.ones(60, dtype=torch.bool)  # All nodes are in the train set
        data.test_mask = torch.ones(60, dtype=torch.bool)  # All nodes are also in the test set
        data.val_mask = torch.ones(60, dtype=torch.bool)
        return [data]


class LCC(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 16
        self.num_classes = 3
        self.num_features = 1
        self.num_nodes = 10
        self.graph_class = False

    def makedata(self):
        generated = False
        while not generated:
            graphs = []
            labels = []
            i = 0
            while i < 6:
                size = 10
                nx_g = nx.random_degree_sequence_graph([3] * size)
                if nx.is_connected(nx_g):
                    i += 1
                    data = from_networkx(nx_g)
                    lbls = [0] * size
                    for n in range(size):
                        edges = 0
                        nbs = [int(nb) for nb in data.edge_index[1][data.edge_index[0] == n]]
                        for nb1 in nbs:
                            for nb2 in nbs:
                                if torch.logical_and(data.edge_index[0] == nb1, data.edge_index[1] == nb2).any():
                                    edges += 1
                        lbls[n] = int(edges / 2)
                    data.y = torch.tensor(lbls)
                    labels.extend(lbls)
                    data = self.addports(data)
                    data = self.makefeatures(data)
                    graphs.append(data)
            generated = labels.count(0) >= 10 and labels.count(1) >= 10 and labels.count(
                2) >= 10  # Ensure the dataset is somewhat balanced

        return graphs


class FourCycles(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.p = 4
        self.hidden_units = 16
        self.num_classes = 2
        self.num_features = 1
        self.num_nodes = 4 * self.p
        self.graph_class = True

    def gen_graph(self, p):
        edge_index = None
        for i in range(p):
            e = torch.tensor([[i, p + i, 2 * p + i, 3 * p + i], [2 * p + i, 3 * p + i, i, p + i]], dtype=torch.long)
            if edge_index is None:
                edge_index = e
            else:
                edge_index = torch.cat([edge_index, e], dim=-1)
        top = np.zeros((p * p,))
        perm = np.random.permutation(range(p))
        for i, t in enumerate(perm):
            top[i * p + t] = 1
        bottom = np.zeros((p * p,))
        perm = np.random.permutation(range(p))
        for i, t in enumerate(perm):
            bottom[i * p + t] = 1
        for i, bit in enumerate(top):
            if bit:
                e = torch.tensor([[i // p, p + i % p], [p + i % p, i // p]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
        for i, bit in enumerate(bottom):
            if bit:
                e = torch.tensor([[2 * p + i // p, 3 * p + i % p], [3 * p + i % p, 2 * p + i // p]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
        return Data(edge_index=edge_index, num_nodes=self.num_nodes), any(np.logical_and(top, bottom))

    def makedata(self):
        size = 25
        p = self.p
        trues = []
        falses = []
        while len(trues) < size or len(falses) < size:
            data, label = self.gen_graph(p)
            data = self.makefeatures(data)
            data = self.addports(data)
            data.y = label
            if label and len(trues) < size:
                trues.append(data)
            elif not label and len(falses) < size:
                falses.append(data)
        return trues + falses


class SkipCircles(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.hidden_units = 32
        self.num_classes = 10  # num skips
        self.num_features = 1
        self.num_nodes = 41
        self.graph_class = True
        self.makedata()

    def makedata(self):
        size = self.num_nodes
        skips = [2, 3, 4, 5, 6, 9, 11, 12, 13, 16]
        graphs = []
        for s, skip in enumerate(skips):
            edge_index = torch.tensor([[0, size - 1], [size - 1, 0]], dtype=torch.long)
            for i in range(size - 1):
                e = torch.tensor([[i, i + 1], [i + 1, i]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
            for i in range(size):
                e = torch.tensor([[i, i], [(i - skip) % size, (i + skip) % size]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
            data = Data(edge_index=edge_index, num_nodes=self.num_nodes)
            data = self.makefeatures(data)
            data = self.addports(data)
            data.y = torch.tensor(s)
            graphs.append(data)

        return graphs



def log_loaded_dataset(dataset, format, name):
    logging.info(f"[*] Loaded dataset '{name}' from '{format}':")
    logging.info(f"  {dataset.data}")
    logging.info(f"  undirected: {dataset[0].is_undirected()}")
    logging.info(f"  num graphs: {len(dataset)}")

    total_num_nodes = 0
    if hasattr(dataset.data, 'num_nodes'):
        total_num_nodes = dataset.data.num_nodes
    elif hasattr(dataset.data, 'x'):
        total_num_nodes = dataset.data.x.size(0)
    logging.info(f"  avg num_nodes/graph: "
                 f"{total_num_nodes // len(dataset)}")
    logging.info(f"  num node features: {dataset.num_node_features}")
    logging.info(f"  num edge features: {dataset.num_edge_features}")
    if hasattr(dataset, 'num_tasks'):
        logging.info(f"  num tasks: {dataset.num_tasks}")

    if hasattr(dataset.data, 'y') and dataset.data.y is not None:
        if isinstance(dataset.data.y, list):
            # A special case for ogbg-code2 dataset.
            logging.info(f"  num classes: n/a")
        elif dataset.data.y.numel() == dataset.data.y.size(0) and \
                torch.is_floating_point(dataset.data.y):
            logging.info(f"  num classes: (appears to be a regression task)")
        else:
            logging.info(f"  num classes: {dataset.num_classes}")
    elif hasattr(dataset.data, 'train_edge_label') or hasattr(dataset.data, 'edge_label'):
        # Edge/link prediction task.
        if hasattr(dataset.data, 'train_edge_label'):
            labels = dataset.data.train_edge_label  # Transductive link task
        else:
            labels = dataset.data.edge_label  # Inductive link task
        if labels.numel() == labels.size(0) and \
                torch.is_floating_point(labels):
            logging.info(f"  num edge classes: (probably a regression task)")
        else:
            logging.info(f"  num edge classes: {len(torch.unique(labels))}")

    ## Show distribution of graph sizes.
    # graph_sizes = [d.num_nodes if hasattr(d, 'num_nodes') else d.x.shape[0]
    #                for d in dataset]
    # hist, bin_edges = np.histogram(np.array(graph_sizes), bins=10)
    # logging.info(f'   Graph size distribution:')
    # logging.info(f'     mean: {np.mean(graph_sizes)}')
    # for i, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
    #     logging.info(
    #         f'     bin {i}: [{start:.2f}, {end:.2f}]: '
    #         f'{hist[i]} ({hist[i] / hist.sum() * 100:.2f}%)'
    #     )

def load_pretrained_gnn(cfg) -> Optional[GraphGymModule]:
    if cfg.posenc_GPSE.enable:
        assert cfg.posenc_GPSE.model_dir is not None
        return load_pretrained_gpse(cfg)
    else:
        return None, lambda: None


def load_pretrained_gpse(cfg) -> Optional[GraphGymModule]:
    if cfg.posenc_GPSE.model_dir is None:
        return None, lambda: None

    logging.info("[*] Setting up GPSE")
    path = cfg.posenc_GPSE.model_dir
    logging.info(f"    Loading pre-trained weights from {path}")
    model_state = torch.load(path, map_location="cpu")["model_state"]
    # Input dimension of the first module in the model weights
    cfg.share.pt_dim_in = dim_in = model_state[list(model_state)[0]].shape[1]
    logging.info(f"    Input dim of the pre-trained model: {dim_in}")
    # Hidden (representation) dimension and final output dimension
    if cfg.posenc_GPSE.gnn_cfg.head.startswith("inductive_hybrid"):
        # Hybrid head dimension inference
        cfg.share.num_graph_targets = model_state[list(model_state)[-1]].shape[0]
        node_head_bias_name = [
            i for i in model_state
            if i.startswith("model.post_mp.node_post_mp")][-1]
        if cfg.posenc_GPSE.gnn_cfg.head.endswith("multi"):
            head_idx = int(
                node_head_bias_name.split("node_post_mps.")[1].split(".model")[0])
            dim_out = head_idx + 1
        else:
            dim_out = model_state[node_head_bias_name].shape[0]
        cfg.share.num_node_targets = dim_out
        logging.info(f"    Graph emb outdim: {cfg.share.num_graph_targets}")
    elif cfg.posenc_GPSE.gnn_cfg.head == "inductive_node_multi":
        dim_out = len([
            1 for i in model_state
            if ("layer_post_mp" in i) and ("layer.model.weight" in i)
        ])
    else:
        dim_out = model_state[list(model_state)[-2]].shape[0]
    if cfg.posenc_GPSE.use_repr:
        cfg.share.pt_dim_out = cfg.posenc_GPSE.gnn_cfg.dim_inner
    else:
        cfg.share.pt_dim_out = dim_out
    logging.info(f"    Outdim of the pre-trained model: {cfg.share.pt_dim_out}")

    # HACK: Temporarily setting global config to default and overwrite GNN
    # configs using the ones from GPSE. Currently, there is no easy way to
    # repurpose the GraphGymModule to build a model using a specified cfg other
    # than completely overwriting the global cfg. [PyG v2.1.0]
    orig_gnn_cfg = CN(cfg.gnn.copy())
    orig_dataset_cfg = CN(cfg.dataset.copy())
    orig_model_cfg = CN(cfg.model.copy())
    plain_cfg = CN()
    set_cfg(plain_cfg)
    # Temporarily replacing the GNN config with the pre-trained GNN predictor
    cfg.gnn = cfg.posenc_GPSE.gnn_cfg
    # Resetting dataset config for bypassing the encoder settings
    cfg.dataset = plain_cfg.dataset
    # Resetting model config to make sure GraphGymModule uses the default GNN
    cfg.model = plain_cfg.model
    logging.info(f"Setting up GPSE using config:\n{cfg.posenc_GPSE.dump()}")

    # Construct model using the patched config and load trained weights
    model = GraphGymModule(dim_in, dim_out, cfg)
    model.load_state_dict(model_state)
    # Set the final linear layer to identity if we want to use the hidden repr
    if cfg.posenc_GPSE.use_repr:
        if cfg.posenc_GPSE.repr_type == "one_layer_before":
            model.model.post_mp.layer_post_mp.model[-1] = torch.nn.Identity()
        elif cfg.posenc_GPSE.repr_type == "no_post_mp":
            model.model.post_mp = IdentityHead()
        else:
            raise ValueError(f"Unknown repr_type {cfg.posenc_GPSE.repr_type!r}")
    model.eval()
    device = get_device(cfg.posenc_GPSE.accelerator, cfg.accelerator)
    model.to(device)
    logging.info(f"Pre-trained model constructed:\n{model}")

    # HACK: Construct bounded function to recover the original configurations
    # to be called right after the pre_transform_in_memory call with
    # compute_posenc_stats is done. This is necessary because things inside
    # GrapyGymModule checks for global configs to determine the behavior for
    # things like forward. To FIX this in the future, need to seriously
    # make sure modules like this store the fixed value at __init__, instead of
    # dynamically looking up configs at runtime.
    def _recover_orig_cfgs():
        cfg.gnn = orig_gnn_cfg
        cfg.dataset = orig_dataset_cfg
        cfg.model = orig_model_cfg

        # Release pretrained model from CUDA memory
        model.to("cpu")
        torch.cuda.empty_cache()

    return model, _recover_orig_cfgs

def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss  # in bytes

@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir):
    """
    Master loader that controls loading of all datasets, overshadowing execution
    of any default GraphGym dataset loader. Default GraphGym dataset loader are
    instead called from this function, the format keywords `PyG` and `OGB` are
    reserved for these default GraphGym loaders.

    Custom transforms and dataset splitting is applied to each loaded dataset.

    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset

    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """
    if format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, pyg_dataset_id)

        if pyg_dataset_id == 'GNNBenchmarkDataset':
            dataset = preformat_GNNBenchmarkDataset(dataset_dir, name)

        elif pyg_dataset_id == 'MalNetTiny':
            dataset = preformat_MalNetTiny(dataset_dir, feature_set=name)

        elif pyg_dataset_id == 'Planetoid':
            dataset = Planetoid(dataset_dir, name)

        elif pyg_dataset_id == 'TUDataset':
            dataset = preformat_TUDataset(dataset_dir, name)

        elif pyg_dataset_id == 'VOCSuperpixels':
            dataset = preformat_VOCSuperpixels(dataset_dir, name,
                                               cfg.dataset.slic_compactness)

        elif pyg_dataset_id == 'COCOSuperpixels':
            dataset = preformat_COCOSuperpixels(dataset_dir, name,
                                                cfg.dataset.slic_compactness)


        elif pyg_dataset_id == 'WikipediaNetwork':
            if name == 'crocodile':
                raise NotImplementedError(f"crocodile not implemented yet")
            dataset = WikipediaNetwork(dataset_dir, name)

        elif pyg_dataset_id == 'ZINC':
            dataset = preformat_ZINC(dataset_dir, name)
            
        elif pyg_dataset_id == 'AQSOL':
            dataset = preformat_AQSOL(dataset_dir, name)

        elif pyg_dataset_id == 'SyntheticCounting':
            dataset = preformat_Counting(dataset_dir, name)

        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

    elif format == 'Synthetic':
        dataset = preformat_syntheic(dataset_dir, name)

    # GraphGym default loader for Pytorch Geometric datasets
    elif format == 'PyG':
        dataset = load_pyg(name, dataset_dir)

    elif format == 'OGB':
        if name.startswith('ogbg'):
            dataset = preformat_OGB_Graph(dataset_dir, name.replace('_', '-'))

        elif name.startswith('PCQM4Mv2-'):
            subset = name.split('-', 1)[1]
            dataset = preformat_OGB_PCQM4Mv2(dataset_dir, subset)

        elif name.startswith('peptides-'):
            dataset = preformat_Peptides(dataset_dir, name)
        elif name.startswith('ogbn'):
            dataset = preformat_ogbn(dataset_dir, name)
        ### Link prediction datasets.
        elif name.startswith('ogbl-'):
            # GraphGym default loader.
            dataset = load_ogb(name, dataset_dir)
            # OGB link prediction datasets are binary classification tasks,
            # however the default loader creates float labels => convert to int.
            def convert_to_int(ds, prop):
                tmp = getattr(ds.data, prop).int()
                set_dataset_attr(ds, prop, tmp, len(tmp))
            convert_to_int(dataset, 'train_edge_label')
            convert_to_int(dataset, 'val_edge_label')
            convert_to_int(dataset, 'test_edge_label')

        elif name.startswith('PCQM4Mv2Contact-'):
            dataset = preformat_PCQM4Mv2Contact(dataset_dir, name)

        else:
            raise ValueError(f"Unsupported OGB(-derived) dataset: {name}")
    else:
        raise ValueError(f"Unknown data format: {format}")
    log_loaded_dataset(dataset, format, name)

    # Precompute necessary statistics for positional encodings.
    pe_enabled_list = []




    for key, pecfg in cfg.items():
        print(pecfg)
        if (key.startswith(('posenc_', 'graphenc_')) and pecfg.enable
                and key != "posenc_GPSE"):  # GPSE handled separately
            pe_name = key.split('_', 1)[1]
            pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel'):
                # Generate kernel times if functional snippet is set.
                if pecfg.kernel.times_func:
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                logging.info(f"Parsed {pe_name} PE kernel times / steps: "
                             f"{pecfg.kernel.times}")



    if pe_enabled_list:
        start = time.perf_counter()
        logging.info(f"Precomputing Positional Encoding statistics: "
                     f"{pe_enabled_list} for all graphs...")
        # Estimate directedness based on 10 graphs to save time.
        is_undirected = all(d.is_undirected() for d in dataset[:10])
        logging.info(f"  ...estimated to be undirected: {is_undirected}")
        if not cfg.dataset.pe_transform_on_the_fly:
            gc.collect()  # Perform garbage collection to avoid measuring old allocations
            mem_before = get_memory_usage()

            pre_transform_in_memory(dataset,
                                    partial(compute_posenc_stats,
                                            pe_types=pe_enabled_list,
                                            is_undirected=is_undirected,
                                            cfg=cfg,
                                            ),
                                    show_progress=True,
                                    cfg=cfg,
                                    posenc_mode=True
                                    )
            mem_after = get_memory_usage()

            # Calculate memory usage
            mem_used = mem_after - mem_before

            logging.info(f"CPU memory used: {mem_used / (1024 ** 2):.2f} MB")

            elapsed = time.perf_counter() - start
            timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                      + f'{elapsed:.2f}'[-3:]
            logging.info(f"Done! Took {timestr}")
        else:
            warnings.warn('PE transform on the fly to save memory consumption; experimental, please only use for RWSE/RWPSE')
            pe_transform = ComputePosencStat(pe_types=pe_enabled_list,
                                             is_undirected=is_undirected,
                                             cfg=cfg
                                             )
            if dataset.transform is None:
                dataset.transform = pe_transform
            else:
                print(dataset.transform)
                print(pe_transform)
                dataset.transform = T.compose([pe_transform, dataset.transform])

    expand = cfg.prep.get("exp", False)
    if expand:
        for j in range(cfg.prep.exp_count):
            start = time.perf_counter()
            logging.info(f"Adding expander edges (round {j}) ...")
            pre_transform_in_memory(dataset,
                                    partial(generate_random_expander,
                                            degree=cfg.prep.exp_deg,
                                            algorithm=cfg.prep.exp_algorithm,
                                            rng=None,
                                            max_num_iters=cfg.prep.exp_max_num_iters,
                                            exp_index=j),
                                    show_progress=True
                                    )
            elapsed = time.perf_counter() - start
            timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                      + f'{elapsed:.2f}'[-3:]
            logging.info(f"Done! Took {timestr}")

    if name == 'ogbn-arxiv' or name == 'ogbn-proteins':
        return dataset
    # Set standard dataset train/val/test splits
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')

    # Verify or generate dataset train/val/test splits
    prepare_splits(dataset)

    # Precompute in-degree histogram if needed for PNAConv.
    if cfg.gt.layer_type.startswith('PNA') and len(cfg.gt.pna_degrees) == 0:
        cfg.gt.pna_degrees = compute_indegree_histogram(
            dataset[dataset.data['train_graph_index']])
        # print(f"Indegrees: {cfg.gt.pna_degrees}")
        # print(f"Avg:{np.mean(cfg.gt.pna_degrees)}")

    if cfg.posenc_GPSE.enable:
        precompute_gpse(cfg, dataset)
    return dataset


def gpse_io(
    dataset,
    mode: str = "save",
    name: Optional[str] = None,
    tag: Optional[str] = None,
    auto_download: bool = True,
):
    assert tag, "Please provide a tag for saving/loading GPSE (e.g., '1.0')"
    pse_dir = dataset.processed_dir
    gpse_data_path = osp.join(pse_dir, f"gpse_{tag}_data.pt")
    gpse_slices_path = osp.join(pse_dir, f"gpse_{tag}_slices.pt")

    def maybe_download_gpse():
        is_complete = osp.isfile(gpse_data_path) and osp.isfile(gpse_slices_path)
        if is_complete or not auto_download:
            return

        if name is None:
            raise ValueError("Please specify the dataset name for downloading.")

        if tag != "1.0":
            raise ValueError(f"Invalid tag {tag!r}, currently only support '1.0")
        # base_url = "https://sandbox.zenodo.org/record/1219850/files/"  # 1.0.dev
        base_url = "https://zenodo.org/record/8145344/files/"  # 1.0
        fname = f"{name}_{tag}.zip"
        url = urljoin(base_url, fname)
        save_path = osp.join(pse_dir, fname)

        # Stream download
        with requests.get(url, stream=True) as r:
            if r.ok:
                total_size_in_bytes = int(r.headers.get("content-length", 0))
                pbar = tqdm(
                    total=total_size_in_bytes,
                    unit="iB",
                    unit_scale=True,
                    desc=f"Downloading {url}",
                )
                with open(save_path, "wb") as file:
                    for data in r.iter_content(1024):
                        pbar.update(len(data))
                        file.write(data)
                pbar.close()

            else:
                meta_url = base_url.replace("/record/", "/api/records/")
                meta_url = meta_url.replace("/files/", "")
                meta_r = requests.get(meta_url)
                if meta_r.ok:
                    files = meta_r.json()["files"]
                    opts = [i["key"].rsplit(".zip")[0] for i in files]
                else:
                    opts = []

                opts_str = "\n".join(sorted(opts))
                raise requests.RequestException(
                    f"Fail to download from {url} ({r!r}). Available options "
                    f"for {tag=!r} are:\n{opts_str}",
                )

        # Unzip files and cleanup
        logging.info(f"Extracting {save_path}")
        with zipfile.ZipFile(save_path, "r") as f:
            f.extractall(pse_dir)
        os.remove(save_path)

    if mode == "save":
        torch.save(dataset.data.pestat_GPSE, gpse_data_path)
        torch.save(dataset.slices["pestat_GPSE"], gpse_slices_path)
        logging.info(f"Saved pre-computed GPSE ({tag}) to {pse_dir}")

    elif mode == "load":
        maybe_download_gpse()
        dataset.data.pestat_GPSE = torch.load(gpse_data_path, map_location="cpu")
        dataset.slices["pestat_GPSE"] = torch.load(gpse_slices_path, map_location="cpu")
        logging.info(f"Loaded pre-computed GPSE ({tag}) from {pse_dir}")

    else:
        raise ValueError(f"Unknown io mode {mode!r}.")


@torch.no_grad()
def precompute_gpse(cfg, dataset):
    dataset_name = f"{cfg.dataset.format}-{cfg.dataset.name}"
    tag = cfg.posenc_GPSE.tag
    if cfg.posenc_GPSE.from_saved:
        gpse_io(dataset, "load", name=dataset_name, tag=tag)
        cfg.share.pt_dim_out = dataset.data.pestat_GPSE.shape[1]
        return

    # Load GPSE model and prepare bounded method to recover original configs
    gpse_model, _recover_orig_cfgs = load_pretrained_gpse(cfg)

    # Temporarily replace the transformation
    orig_dataset_transform = dataset.transform
    dataset.transform = None
    if cfg.posenc_GPSE.virtual_node:
        dataset.transform = VirtualNodePatchSingleton()

    # Remove split indices, to be recovered at the end of the precomputation
    tmp_store = {}
    for name in ["train_mask", "val_mask", "test_mask", "train_graph_index",
                 "val_graph_index", "test_graph_index", "train_edge_index",
                 "val_edge_index", "test_edge_index"]:
        if (name in dataset.data) and (dataset.slices is None
                                       or name in dataset.slices):
            tmp_store_data = dataset.data.pop(name)
            tmp_store_slices = dataset.slices.pop(name) if dataset.slices else None
            tmp_store[name] = (tmp_store_data, tmp_store_slices)

    loader = DataLoader(dataset, batch_size=cfg.posenc_GPSE.loader.batch_size,
                        shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=True, persistent_workers=cfg.num_workers > 0)

    # Batched GPSE precomputation loop
    data_list = []
    curr_idx = 0
    pbar = trange(len(dataset), desc="Pre-computing GPSE")
    tic = time.perf_counter()
    for batch in loader:
        batch_out, batch_ptr = gpse_process_batch(gpse_model, batch)

        batch_out = batch_out.to("cpu", non_blocking=True)
        # Need to wait for batch_ptr to finish transfering so that start and
        # end indices are ready to use
        batch_ptr = batch_ptr.to("cpu", non_blocking=False)

        for start, end in zip(batch_ptr[:-1], batch_ptr[1:]):
            data = dataset.get(curr_idx)
            if cfg.posenc_GPSE.virtual_node:
                end = end - 1
            data.pestat_GPSE = batch_out[start:end]
            data_list.append(data)
            curr_idx += 1

        pbar.update(len(batch_ptr) - 1)
    pbar.close()

    # Collate dataset and reset indicies and data list
    dataset.transform = orig_dataset_transform
    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)

    # Recover split indices
    for name, (tmp_store_data, tmp_store_slices) in tmp_store.items():
        dataset.data[name] = tmp_store_data
        if tmp_store_slices is not None:
            dataset.slices[name] = tmp_store_slices
    dataset._data_list = None

    if cfg.posenc_GPSE.save:
        gpse_io(dataset, "save", tag=tag)

    timestr = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - tic))
    logging.info(f"Finished GPSE pre-computation, took {timestr}")

    # Release resource and recover original configs
    del gpse_model
    torch.cuda.empty_cache()
    _recover_orig_cfgs()

def add_pe_transform_to_dataset(format, name, dataset_dir, pe_transform=None):
    if format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, pyg_dataset_id)

        if pyg_dataset_id == 'GNNBenchmarkDataset':
            dataset = preformat_GNNBenchmarkDataset(dataset_dir, name)

        elif pyg_dataset_id == 'MalNetTiny':
            dataset = preformat_MalNetTiny(dataset_dir, feature_set=name)

        elif pyg_dataset_id == 'Planetoid':
            dataset = Planetoid(dataset_dir, name)

        elif pyg_dataset_id == 'TUDataset':
            dataset = preformat_TUDataset(dataset_dir, name)

        elif pyg_dataset_id == 'VOCSuperpixels':
            dataset = preformat_VOCSuperpixels(dataset_dir, name,
                                               cfg.dataset.slic_compactness)

        elif pyg_dataset_id == 'COCOSuperpixels':
            dataset = preformat_COCOSuperpixels(dataset_dir, name,
                                                cfg.dataset.slic_compactness)


        elif pyg_dataset_id == 'WikipediaNetwork':
            if name == 'crocodile':
                raise NotImplementedError(f"crocodile not implemented yet")
            dataset = WikipediaNetwork(dataset_dir, name)

        elif pyg_dataset_id == 'ZINC':
            dataset = preformat_ZINC(dataset_dir, name)

        elif pyg_dataset_id == 'AQSOL':
            dataset = preformat_AQSOL(dataset_dir, name)

        elif pyg_dataset_id == 'SyntheticCounting':
            dataset = preformat_Counting(dataset_dir, name)

        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

    elif format == 'Synthetic':
        pyg_dataset_id = format.split('-', 1)[1]
        dataset = preformat_syntheic(dataset_dir, pyg_dataset_id)

    # GraphGym default loader for Pytorch Geometric datasets
    elif format == 'PyG':
        dataset = load_pyg(name, dataset_dir)

    elif format == 'OGB':
        if name.startswith('ogbg'):
            dataset = preformat_OGB_Graph(dataset_dir, name.replace('_', '-'))

        elif name.startswith('PCQM4Mv2-'):
            subset = name.split('-', 1)[1]
            dataset = preformat_OGB_PCQM4Mv2(dataset_dir, subset)

        elif name.startswith('peptides-'):
            dataset = preformat_Peptides(dataset_dir, name)

        ### Link prediction datasets.
        elif name.startswith('ogbl-'):
            # GraphGym default loader.
            dataset = load_ogb(name, dataset_dir)
            # OGB link prediction datasets are binary classification tasks,
            # however the default loader creates float labels => convert to int.
            def convert_to_int(ds, prop):
                tmp = getattr(ds.data, prop).int()
                set_dataset_attr(ds, prop, tmp, len(tmp))
            convert_to_int(dataset, 'train_edge_label')
            convert_to_int(dataset, 'val_edge_label')
            convert_to_int(dataset, 'test_edge_label')

        elif name.startswith('PCQM4Mv2Contact-'):
            dataset = preformat_PCQM4Mv2Contact(dataset_dir, name)

        else:
            raise ValueError(f"Unsupported OGB(-derived) dataset: {name}")
    else:
        raise ValueError(f"Unknown data format: {format}")
    log_loaded_dataset(dataset, format, name)



def compute_indegree_histogram(dataset):
    """Compute histogram of in-degree of nodes needed for PNAConv.

    Args:
        dataset: PyG Dataset object

    Returns:
        List where i-th value is the number of nodes with in-degree equal to `i`
    """
    from torch_geometric.utils import degree

    deg = torch.zeros(1000, dtype=torch.long)
    max_degree = 0
    for data in dataset:
        d = degree(data.edge_index[1],
                   num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, d.max().item())
        deg += torch.bincount(d, minlength=deg.numel())
    return deg.numpy().tolist()[:max_degree + 1]

from torch_geometric.data import InMemoryDataset, Data

class SyntheticGraphDataset(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None, pre_transform=None, split = "train"):
        super().__init__(root, transform, pre_transform)
        self.split = split
        if data_list is not None:
            self.data, self.slices = self.collate(data_list)
        else:
            self.data, self.slices = None, None

    def _download(self):
        pass  # No download needed

    def _process(self):
        pass  # No processing needed

class SyntheticNodeDataset(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None, pre_transform=None):
        self.data_list = data_list
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.collate(self.data_list)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass


def preformat_syntheic(dataset_dir, name):
    """Load and preformat datasets from Synthetic Datasets of DropGNN.
        https://arxiv.org/pdf/2111.06283
    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the Synthetic Datasets

    Returns:
        PyG dataset object
    """
    dataset = SkipCircles()
    if name == "skipcircles":
        dataset = SkipCircles()
    elif name == "triangles":
        dataset = Triangles()
    elif name == "limitsone":
        dataset = LimitsOne()
    elif name == "limitstwo":
        dataset = LimitsTwo()
    elif name == "fourcycles":
        dataset = FourCycles()


    if name in ['skipcircles', 'fourcycles']:
        dataset_splits = ['train', 'val', 'test']
        datasets = []
        for split in dataset_splits:
            data_list = dataset.makedata()  # Randomized data for each split

            syn_dataset = SyntheticGraphDataset(root=dataset_dir, split=split, data_list=data_list)
            datasets.append(syn_dataset)
        dataset = join_dataset_splits(
            datasets
        )
        return dataset
    else:
        syn_dataset = SyntheticNodeDataset(root=dataset_dir, data_list=dataset.makedata())
        return syn_dataset




def preformat_GNNBenchmarkDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's GNNBenchmarkDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """
    tf_list = []
    if name in ['MNIST', 'CIFAR10']:
        tf_list = [concat_x_and_pos]  # concat pixel value and pos. coordinate
        tf_list.append(partial(typecast_x, type_str='float'))
    else:
        ValueError(f"Loading dataset '{name}' from "
                   f"GNNBenchmarkDataset is not supported.")

    dataset_dir = dataset_dir.replace("/GNNBenchmarkDataset", "")
    dataset = join_dataset_splits(
        [GNNBenchmarkDataset(root=dataset_dir, name=name, split=split)
         for split in ['train', 'val', 'test']]
    )
    pre_transform_in_memory(dataset, T.Compose(tf_list))

    return dataset




def preformat_MalNetTiny(dataset_dir, feature_set):
    """Load and preformat Tiny version (5k graphs) of MalNet

    Args:
        dataset_dir: path where to store the cached dataset
        feature_set: select what node features to precompute as MalNet
            originally doesn't have any node nor edge features

    Returns:
        PyG dataset object
    """
    if feature_set in ['none', 'Constant']:
        tf = T.Constant()
    elif feature_set == 'OneHotDegree':
        tf = T.OneHotDegree()
    elif feature_set == 'LocalDegreeProfile':
        tf = T.LocalDegreeProfile()
    else:
        raise ValueError(f"Unexpected transform function: {feature_set}")

    dataset = MalNetTiny(dataset_dir)
    dataset.name = 'MalNetTiny'
    logging.info(f'Computing "{feature_set}" node features for MalNetTiny.')
    pre_transform_in_memory(dataset, tf)

    split_dict = dataset.get_idx_split()
    dataset.split_idxs = [split_dict['train'],
                          split_dict['valid'],
                          split_dict['test']]

    return dataset


def preformat_OGB_Graph(dataset_dir, name):
    """Load and preformat OGB Graph Property Prediction datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific OGB Graph dataset

    Returns:
        PyG dataset object
    """
    dataset = PygGraphPropPredDataset(name=name, root=dataset_dir)
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ['train', 'valid', 'test']]

    if name == 'ogbg-ppa':
        # ogbg-ppa doesn't have any node features, therefore add zeros but do
        # so dynamically as a 'transform' and not as a cached 'pre-transform'
        # because the dataset is big (~38.5M nodes), already taking ~31GB space
        def add_zeros(data):
            data.x = torch.zeros(data.num_nodes, dtype=torch.long)
            return data
        dataset.transform = add_zeros
    elif name == 'ogbg-code2':
        from grit.loader.ogbg_code2_utils import idx2vocab, \
            get_vocab_mapping, augment_edge, encode_y_to_arr
        num_vocab = 5000  # The number of vocabulary used for sequence prediction
        max_seq_len = 5  # The maximum sequence length to predict

        seq_len_list = np.array([len(seq) for seq in dataset.data.y])
        logging.info(f"Target sequences less or equal to {max_seq_len} is "
            f"{np.sum(seq_len_list <= max_seq_len) / len(seq_len_list)}")

        # Building vocabulary for sequence prediction. Only use training data.
        vocab2idx, idx2vocab_local = get_vocab_mapping(
            [dataset.data.y[i] for i in s_dict['train']], num_vocab)
        logging.info(f"Final size of vocabulary is {len(vocab2idx)}")
        idx2vocab.extend(idx2vocab_local)  # Set to global variable to later access in CustomLogger

        # Set the transform function:
        # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
        # encode_y_to_arr: add y_arr to PyG data object, indicating the array repres
        dataset.transform = T.Compose(
            [augment_edge,
             lambda data: encode_y_to_arr(data, vocab2idx, max_seq_len)])

        # Subset graphs to a maximum size (number of nodes) limit.
        pre_transform_in_memory(dataset, partial(clip_graphs_to_size,
                                                 size_limit=1000))

    return dataset


def preformat_OGB_PCQM4Mv2(dataset_dir, name):
    """Load and preformat PCQM4Mv2 from OGB LSC.

    OGB-LSC provides 4 data index splits:
    2 with labeled molecules: 'train', 'valid' meant for training and dev
    2 unlabeled: 'test-dev', 'test-challenge' for the LSC challenge submission

    We will take random 150k from 'train' and make it a validation set and
    use the original 'valid' as our testing set.

    Note: PygPCQM4Mv2Dataset requires rdkit

    Args:
        dataset_dir: path where to store the cached dataset
        name: select 'subset' or 'full' version of the training set

    Returns:
        PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary.
        from ogb.lsc import PygPCQM4Mv2Dataset
    except Exception as e:
        logging.error('ERROR: Failed to import PygPCQM4Mv2Dataset, '
                      'make sure RDKit is installed.')
        raise e


    dataset = PygPCQM4Mv2Dataset(root=dataset_dir)
    split_idx = dataset.get_idx_split()

    rng = default_rng(seed=42)
    train_idx = rng.permutation(split_idx['train'].numpy())
    train_idx = torch.from_numpy(train_idx)

    # Leave out 150k graphs for a new validation set.
    valid_idx, train_idx = train_idx[:150000], train_idx[150000:]
    if name == 'full':
        split_idxs = [train_idx,  # Subset of original 'train'.
                      valid_idx,  # Subset of original 'train' as validation set.
                      split_idx['valid']  # The original 'valid' as testing set.
                      ]

    elif name == 'subset':
        # Further subset the training set for faster debugging.
        subset_ratio = 0.1
        subtrain_idx = train_idx[:int(subset_ratio * len(train_idx))]
        subvalid_idx = valid_idx[:50000]
        subtest_idx = split_idx['valid']  # The original 'valid' as testing set.

        dataset = dataset[torch.cat([subtrain_idx, subvalid_idx, subtest_idx])]
        data_list = [data for data in dataset]
        dataset._indices = None
        dataset._data_list = data_list
        dataset.data, dataset.slices = dataset.collate(data_list)
        n1, n2, n3 = len(subtrain_idx), len(subvalid_idx), len(subtest_idx)
        split_idxs = [list(range(n1)),
                      list(range(n1, n1 + n2)),
                      list(range(n1 + n2, n1 + n2 + n3))]

    elif name == 'inference':
        split_idxs = [split_idx['valid'],  # The original labeled 'valid' set.
                      split_idx['test-dev'],  # Held-out unlabeled test dev.
                      split_idx['test-challenge']  # Held-out challenge test set.
                      ]

        dataset = dataset[torch.cat(split_idxs)]
        data_list = [data for data in dataset]
        dataset._indices = None
        dataset._data_list = data_list
        dataset.data, dataset.slices = dataset.collate(data_list)
        n1, n2, n3 = len(split_idxs[0]), len(split_idxs[1]), len(split_idxs[2])
        split_idxs = [list(range(n1)),
                      list(range(n1, n1 + n2)),
                      list(range(n1 + n2, n1 + n2 + n3))]
        # Check prediction targets.
        assert(all([not torch.isnan(dataset[i].y)[0] for i in split_idxs[0]]))
        assert(all([torch.isnan(dataset[i].y)[0] for i in split_idxs[1]]))
        assert(all([torch.isnan(dataset[i].y)[0] for i in split_idxs[2]]))

    else:
        raise ValueError(f'Unexpected OGB PCQM4Mv2 subset choice: {name}')
    dataset.split_idxs = split_idxs
    return dataset


def preformat_PCQM4Mv2Contact(dataset_dir, name):
    """Load PCQM4Mv2-derived molecular contact link prediction dataset.

    Note: This dataset requires RDKit dependency!

    Args:
       dataset_dir: path where to store the cached dataset
       name: the type of dataset split: 'shuffle', 'num-atoms'

    Returns:
       PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary
        from grit.loader.dataset.pcqm4mv2_contact import \
            PygPCQM4Mv2ContactDataset, \
            structured_neg_sampling_transform
    except Exception as e:
        logging.error('ERROR: Failed to import PygPCQM4Mv2ContactDataset, '
                      'make sure RDKit is installed.')
        raise e

    split_name = name.split('-', 1)[1]
    dataset = PygPCQM4Mv2ContactDataset(dataset_dir, subset='530k')
    # Inductive graph-level split (there is no train/test edge split).
    s_dict = dataset.get_idx_split(split_name)
    dataset.split_idxs = [s_dict[s] for s in ['train', 'val', 'test']]
    if cfg.dataset.resample_negative:
        dataset.transform = structured_neg_sampling_transform
    return dataset


def preformat_Peptides(dataset_dir, name):
    """Load Peptides dataset, functional or structural.

    Note: This dataset requires RDKit dependency!

    Args:
        dataset_dir: path where to store the cached dataset
        name: the type of dataset split:
            - 'peptides-functional' (10-task classification)
            - 'peptides-structural' (11-task regression)

    Returns:
        PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary.
        from grit.loader.dataset.peptides_functional import \
            PeptidesFunctionalDataset
        from grit.loader.dataset.peptides_structural import \
            PeptidesStructuralDataset
    except Exception as e:
        logging.error('ERROR: Failed to import Peptides dataset class, '
                      'make sure RDKit is installed.')
        raise e

    dataset_type = name.split('-', 1)[1]
    if dataset_type == 'functional':
        dataset = PeptidesFunctionalDataset(dataset_dir)
    elif dataset_type == 'structural':
        dataset = PeptidesStructuralDataset(dataset_dir)
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ['train', 'val', 'test']]
    return dataset


def preformat_TUDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's TUDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """
    if name in ['DD', 'NCI1', 'ENZYMES', 'PROTEINS']:
        func = None
    elif name.startswith('IMDB-') or name == "COLLAB":
        func = T.Constant()
    else:
        ValueError(f"Loading dataset '{name}' from TUDataset is not supported.")
    dataset = TUDataset(dataset_dir, name, pre_transform=func)
    return dataset

def preformat_ogbn(dataset_dir, name):
  if name == 'ogbn-arxiv' or name == 'ogbn-proteins':
    dataset = PygNodePropPredDataset(name=name)
    if name == 'ogbn-arxiv':
      pre_transform_in_memory(dataset, partial(add_reverse_edges))
      if cfg.prep.add_self_loops:
        pre_transform_in_memory(dataset, partial(add_self_loops))
    if name == 'ogbn-proteins':
      pre_transform_in_memory(dataset, partial(move_node_feat_to_x))
      pre_transform_in_memory(dataset, partial(typecast_x, type_str='float'))
    split_dict = dataset.get_idx_split()
    split_dict['val'] = split_dict.pop('valid')
    dataset.split_idx = split_dict
    return dataset


     #  We do not need to store  these separately.
     # storing separatelymight simplify the duplicated logger code in main.py
     # s_dict = dataset.get_idx_split()
     # dataset.split_idxs = [s_dict[s] for s in ['train', 'valid', 'test']]
     # convert the adjacency list to an edge_index list.
     # data = dataset[0]
     # coo = data.adj_t.coo()
     # data is only a deep copy.  Need to write to the dataset object itself.
     # dataset[0].edge_index = torch.stack(coo[:2])
     # del dataset[0]['adj_t'] # remove the adjacency list after the edge_index is created.

     # return dataset
  else:
     ValueError(f"Unknown ogbn dataset '{name}'.")
def preformat_ZINC(dataset_dir, name):
    """Load and preformat ZINC datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: select 'subset' or 'full' version of ZINC

    Returns:
        PyG dataset object
    """
    if name not in ['subset', 'full']:
        raise ValueError(f"Unexpected subset choice for ZINC dataset: {name}")
    dataset = join_dataset_splits(
        [ZINC(root=dataset_dir, subset=(name == 'subset'), split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset

def preformat_Counting(dataset_dir, name):
    """Load and preformat Counting data set used in:
        https://github.com/leichen2018/GNN-Substructure-Counting
    Args:
        dataset_dir: path where the bin file is saved
        name: 'star-dataset1' or 'triangle-dataset2' or 'attributed_triangle-dataset1'
    """
    task, dataset_type = name.split('-')
    dataset = join_dataset_splits(
        [SyntheticCounting(root=dataset_dir, task=task, dataset_type=dataset_type, split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_AQSOL(dataset_dir):
    """Load and preformat AQSOL datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [AQSOL(root=dataset_dir, split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_VOCSuperpixels(dataset_dir, name, slic_compactness):
    """Load and preformat VOCSuperpixels dataset.

    Args:
        dataset_dir: path where to store the cached dataset
    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [VOCSuperpixels(root=dataset_dir, name=name,
                        slic_compactness=slic_compactness,
                        split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_COCOSuperpixels(dataset_dir, name, slic_compactness):
    """Load and preformat COCOSuperpixels dataset.

    Args:
        dataset_dir: path where to store the cached dataset
    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [COCOSuperpixels(root=dataset_dir, name=name,
                         slic_compactness=slic_compactness,
                         split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]

def set_virtual_node(dataset):
    if dataset.transform_list is None:
        dataset.transform_list = []
    dataset.transform_list.append(VirtualNodePatchSingleton())