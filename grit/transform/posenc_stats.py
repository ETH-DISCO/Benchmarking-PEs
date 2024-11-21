from copy import deepcopy
import logging
import math
from scipy.sparse.linalg import expm
import scipy.sparse as sp

import numpy as np

import torch
import torch.nn.functional as F
# from numpy.linalg import eigvals
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj, degree)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add
from functools import partial
from .rrwp import add_full_rrwp
from torch_ppr import page_rank, personalized_page_rank
# from node2vec import Node2Vec
# import networkx as nx
# from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

from ..utils import wl_positional_encoding, adj_mul
import torch_geometric.utils as utils
from attrdict import AttrDict
import torch_geometric
import json
from ..encoder.neuraldrawer.network.preprocessing import preprocess_dataset
from ..encoder.neuraldrawer.network.model import get_model
import os

class DisableLogging:
    def __enter__(self):
        self.original_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.CRITICAL + 1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.getLogger().setLevel(self.original_level)


def append_cols_to_match(U, max_freqs):
    # U is your matrix, and max_freqs is the target number of columns.
    num_cols_to_add = max_freqs - U.shape[1]

    if num_cols_to_add > 0:
        # Create a matrix of zeros with the same number of rows as U and num_cols_to_add columns.
        zeros_matrix = np.zeros((U.shape[0], num_cols_to_add), dtype=np.float32)

        # Concatenate U and zeros_matrix along the column axis.
        U = np.hstack((U, zeros_matrix))

    return U

    return U
def positional_embedding(wl_pos_enc, d_h):
    """Calcuate the pos_enc for each node with vanilla positional encoding in original Transformer model
    wl_pos_enc: wl-based positional encoding
    d_h: dimension of positional encoding

    output: positional encoding for each node in the graph:
    size: num_nodes * d_h
    """
    assert d_h % 2 == 0
    max_len = wl_pos_enc.size(0)
    pe = torch.zeros(max_len, d_h)
    div_term = torch.exp(torch.arange(0, d_h, 2).float() * -(math.log(10000.0) / d_h))
    pe[:, 0::2] = torch.sin(wl_pos_enc.unsqueeze(1) * div_term)
    pe[:, 1::2] = torch.cos(wl_pos_enc.unsqueeze(1) * div_term)
    return pe

def compute_posenc_stats(data, pe_types, is_undirected, cfg):
    """Precompute positional encodings for the given graph.
    Supported PE statistics to precompute, selected by `pe_types`:
    'LapPE': Laplacian eigen-decomposition.
    'RWSE': Random walk landing probabilities (diagonals of RW matrices).
    'HKfullPE': Full heat kernels and their diagonals. (NOT IMPLEMENTED)
    'HKdiagSE': Diagonals of heat kernel diffusion.
    'ElstaticSE': Kernel based on the electrostatic interaction between nodes.
    'RRWP': Relative Random Walk Probabilities PE (Ours, for GRIT)
    'SVD': SVD decomposition of the adjacency matrix (Edge Augmented Transformer)
    'GPSE':Graph Positional and Structural Encoder, ICML 2024
    'PPR': Personalized PageRank
    'NODE2VEC': Node2Vec
    'WLPE': WL test based positional encodings for the given graph
    'GCKN': Graph Kernel based positional encodings for the given graph
    'RWDIFF': LSPE: Geometric Diffusion with Random Walk Probabilities
    Args:
        data: PyG graph
        pe_types: Positional encoding types to precompute statistics for.
            This can also be a combination, e.g. 'eigen+rw_landing'
        is_undirected: True if the graph is expected to be undirected
        cfg: Main configuration node

    Returns:
        Extended PyG Data object.
    """
    # Verify PE types.

    # cfg.num_nodes = data.num_nodes

    for t in pe_types:
        if t not in ['LapPE', 'EquivStableLapPE', 'SignNet',
                     'RWSE', 'HKdiagSE', 'HKfullPE', 'ElstaticSE','RRWP',
                     'SVD', 'PPR', 'WLPE', 'GCKN', 'RWDIFF', 'GD', 'GPSE']:
            raise ValueError(f"Unexpected PE stats selection {t} in {pe_types}")

    # Basic preprocessing of the input graph.
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    laplacian_norm_type = cfg.posenc_LapPE.eigen.laplacian_norm.lower()
    if laplacian_norm_type == 'none':
        laplacian_norm_type = None
    if is_undirected:
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)

    # Eigen values and vectors.
    evals, evects = None, None
    if 'LapPE' in pe_types or 'EquivStableLapPE' in pe_types:
        # Eigen-decomposition with numpy, can be reused for Heat kernels.
        L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_index, normalization=laplacian_norm_type,
                           num_nodes=N)
        )
        evals, evects = np.linalg.eigh(L.toarray())
        
        if 'LapPE' in pe_types:
            max_freqs=cfg.posenc_LapPE.eigen.max_freqs
            eigvec_norm=cfg.posenc_LapPE.eigen.eigvec_norm
        elif 'EquivStableLapPE' in pe_types:  
            max_freqs=cfg.posenc_EquivStableLapPE.eigen.max_freqs
            eigvec_norm=cfg.posenc_EquivStableLapPE.eigen.eigvec_norm
        
        data.EigVals, data.EigVecs = get_lap_decomp_stats(
            evals=evals, evects=evects,
            max_freqs=max_freqs,
            eigvec_norm=eigvec_norm)

    if 'GCKN' in pe_types:

        edge_attr = data.edge_attr if cfg.posenc_GCKN.use_edge_attr else None
        L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_index, edge_attr, normalization=laplacian_norm_type,
                           num_nodes=N)
        )

        if cfg.posenc_GCKN.method == "diffusion":
            L = expm(-cfg.posenc_GCKN.beta * L)
            evals, evects = np.linalg.eigh(L.toarray())
        elif cfg.posenc_GCKN.method == "pRWSE":
            L = sp.identity(L.shape[0], dtype=L.dtype) - cfg.posenc_GCKN.beta * L
            tmp = L
            for _ in range(cfg.posenc_GCKN.p - 1):
                tmp = tmp.dot(L)
            evals, evects = np.linalg.eigh(tmp.toarray())


        max_freqs = cfg.posenc_GCKN.eigen.max_freqs
        eigvec_norm = cfg.posenc_GCKN.eigen.eigvec_norm

        data.EigVals, data.EigVecs = get_lap_decomp_stats(
            evals=evals, evects=evects,
            max_freqs=max_freqs,
            eigvec_norm=eigvec_norm)


    if 'SVD' in pe_types:
        # Eigen-decomposition on rectified adjacency mtx
        mtx, _ = add_self_loops(undir_edge_index)
        adj_with_self_loop = to_scipy_sparse_matrix(mtx)
        u, s, vh = np.linalg.svd(adj_with_self_loop.toarray())
        data.pos_enc = get_svd_decomp_stats(u, s, vh, cfg.posenc_SVD.eigen.max_freqs, 'L2')

    if 'PPR' in pe_types:
        # PPR enc is directly defined in encoder
        # No eigenvalue decomposition requires here
        n = data.num_nodes
        ppr_matrix = np.zeros((n, n))
        ppr_matrix = personalized_page_rank(edge_index = undir_edge_index, indices = np.arange(n))
        U, Sigma, VT = np.linalg.svd(ppr_matrix.cpu(), full_matrices = True)


        data.pos_enc = append_cols_to_match(U[:, :cfg.posenc_PPR.eigen.max_freqs], cfg.posenc_PPR.eigen.max_freqs)
        # print(data.pos_enc.shape)

    # if 'NODE2VEC' in pe_types:
    #     # print("Dealing with node2vec embeddings ...")
    #     G = to_networkx(data, to_undirected=True)
    #     node2vec = Node2Vec(G, dimensions=cfg.posenc_NODE2VEC.node2vec.dimensions,
    #                            walk_length=cfg.posenc_NODE2VEC.node2vec.walk_length,
    #                            num_walks=cfg.posenc_NODE2VEC.node2vec.num_walks,
    #                            workers=cfg.posenc_NODE2VEC.node2vec.workers,
    #                            quiet=True)
    #     with DisableLogging():
    #         model = node2vec.fit(window=cfg.posenc_NODE2VEC.node2vec.window,
    #                          min_count=cfg.posenc_NODE2VEC.node2vec.min_count,
    #                          batch_words=cfg.posenc_NODE2VEC.node2vec.batch_words,
    #
    #                          )
    #
    #     embeddings = {str(node): model.wv[str(node)] for node in G.nodes()}
    #     embedding_list = [embeddings[str(node)] for node in sorted(G.nodes(), key=lambda x: str(x))]
    #     data.pos_enc = np.vstack(embedding_list)

    if 'WLPE' in pe_types:
        data.pos_enc = positional_embedding(wl_positional_encoding(data),
                                           cfg.posenc_WLPE.dh)

    if 'RWDIFF' in pe_types:
        n = data.num_nodes
        # Geometric diffusion features with Random Walk
        A = to_scipy_sparse_matrix(undir_edge_index, num_nodes=n)  # Get adjacency matrix as a CSR matrix
        Dinv = degree(data.edge_index[0], num_nodes=n).clip(1)
        Dinv = 1.0 / Dinv.numpy()
        Dinv = sp.diags(Dinv, offsets=0, shape=(n, n), format='csr')  # Create a diagonal matrix

        RW = A.dot(Dinv)
        M = RW
        # Iterate
        # nb_pos_enc = cfg.posenc_RWDIFF.pos_enc_dim
        nb_pos_enc = cfg.posenc_RWDIFF.dim_pe
        PE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(nb_pos_enc - 1):
            M_power = M_power @ M  # Use @ for matrix multiplication
            PE.append(torch.from_numpy(M_power.diagonal()).float())
        PE = torch.stack(PE, dim=-1)
        data.pos_enc = PE

    if 'SignNet' in pe_types:
        # Eigen-decomposition with numpy for SignNet.
        norm_type = cfg.posenc_SignNet.eigen.laplacian_norm.lower()
        if norm_type == 'none':
            norm_type = None
        L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_index, normalization=norm_type,
                           num_nodes=N)
        )
        evals_sn, evects_sn = np.linalg.eigh(L.toarray())
        data.eigvals_sn, data.eigvecs_sn = get_lap_decomp_stats(
            evals=evals_sn, evects=evects_sn,
            max_freqs=cfg.posenc_SignNet.eigen.max_freqs,
            eigvec_norm=cfg.posenc_SignNet.eigen.eigvec_norm)

    # Random Walks.
    if 'RWSE' in pe_types:
        kernel_param = cfg.posenc_RWSE.kernel
        if len(kernel_param.times) == 0:
            raise ValueError("List of kernel times required for RWSE")
        rw_landing = get_rw_landing_probs(ksteps=kernel_param.times,
                                          edge_index=data.edge_index,
                                          num_nodes=N)
        data.pestat_RWSE = rw_landing

    # Heat Kernels.
    if 'HKdiagSE' in pe_types or 'HKfullPE' in pe_types:
        # Get the eigenvalues and eigenvectors of the regular Laplacian,
        # if they have not yet been computed for 'eigen'.
        if laplacian_norm_type is not None or evals is None or evects is None:
            L_heat = to_scipy_sparse_matrix(
                *get_laplacian(undir_edge_index, normalizationv=None, num_nodes=N)
            )
            evals_heat, evects_heat = np.linalg.eigh(L_heat.toarray())
        else:
            evals_heat, evects_heat = evals, evects
        evals_heat = torch.from_numpy(evals_heat)
        evects_heat = torch.from_numpy(evects_heat)

        # Get the full heat kernels.
        if 'HKfullPE' in pe_types:
            # The heat kernels can't be stored in the Data object without
            # additional padding because in PyG's collation of the graphs the
            # sizes of tensors must match except in dimension 0. Do this when
            # the full heat kernels are actually used downstream by an Encoder.
            raise NotImplementedError()
            # heat_kernels, hk_diag = get_heat_kernels(evects_heat, evals_heat,
            #                                   kernel_times=kernel_param.times)
            # data.pestat_HKdiagSE = hk_diag
        # Get heat kernel diagonals in more efficient way.
        if 'HKdiagSE' in pe_types:
            kernel_param = cfg.posenc_HKdiagSE.kernel
            if len(kernel_param.times) == 0:
                raise ValueError("Diffusion times are required for heat kernel")
            hk_diag = get_heat_kernels_diag(evects_heat, evals_heat,
                                            kernel_times=kernel_param.times,
                                            space_dim=0)
            data.pestat_HKdiagSE = hk_diag

    # Electrostatic interaction inspired kernel.
    if 'ElstaticSE' in pe_types:
        elstatic = get_electrostatic_function_encoding(undir_edge_index, N)
        data.pestat_ElstaticSE = elstatic

    if 'RRWP' in pe_types:
        param = cfg.posenc_RRWP
        transform = partial(add_full_rrwp,
                            walk_length=param.ksteps,
                            attr_name_abs="rrwp",
                            attr_name_rel="rrwp",
                            add_identity=True,
                            spd=param.spd, # by default False
                            )
        data = transform(data)

    if 'GD' in pe_types:
        pecfg = cfg.posenc_GD
        if pecfg.get("measurement", False):

            model_path = pecfg.model_path
            config_path = pecfg.config_path

            if pecfg.use_embeddings:
                use_embedding = True
            else:
                use_embedding = False

            with open(os.getcwd() + config_path, 'r') as f:

                config = AttrDict(json.load(f))

            eval_model = get_model(config)
            eval_model.load_state_dict(torch.load(os.getcwd() + model_path))
            for param in eval_model.parameters():
                param.requires_grad = False



            with torch.no_grad():
                original_edges = data.edge_index
                data.x_orig = data.x
                data_list = [data]

                data_list[0].edge_index = torch_geometric.utils.to_undirected(data_list[0].edge_index)
                data_list = preprocess_dataset(data_list, config)
                batch = torch_geometric.data.Batch.from_data_list(data_list)

                pred, pos_enc = eval_model(batch, 20, return_layers=True)
                data.edge_index = original_edges

                if use_embedding:
                    pos_enc = pos_enc[-1].detach()
                else:
                    pos_enc = pred.detach()

            data.pos_enc = pos_enc

    ### Deal with two places where
    # if cfg.gt.layer_type == "SAT":
    #     data.degree = compute_degree(data)
    #     node_indices = []
    #     edge_indices = []
    #     edge_attributes = []
    #     indicators = []
    #     edge_index_start = 0
    #
    #     for node_idx in range(data.num_nodes):
    #         sub_nodes, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(
    #             node_idx,
    #             cfg.gt.k_hop,
    #             data.edge_index,
    #             relabel_nodes=True,
    #             num_nodes=data.num_nodes
    #         )
    #         node_indices.append(sub_nodes)
    #         edge_indices.append(sub_edge_index + edge_index_start)
    #         indicators.append(torch.zeros(sub_nodes.shape[0]).fill_(node_idx))
    #         # if self.use_subgraph_edge_attr and graph.edge_attr is not None:
    #         edge_attributes.append(data.edge_attr[edge_mask])  # CHECK THIS DIDN"T BREAK ANYTHING
    #         edge_index_start += len(sub_nodes)
    #
    #     data.subgraph_node_idx = torch.cat(node_indices)
    #     data.subgraph_edge_index = torch.cat(edge_indices, dim=1)
    #     data.subgraph_indicator = torch.cat(indicators)
    #     data.subgraph_edge_attr = torch.cat(edge_attributes)
    #     data.num_subgraph_nodes = len(data.subgraph_edge_index)
    return data

def compute_degree(dataset):
    return 1. / torch.sqrt(1. + utils.degree(dataset.edge_index[0], dataset.num_nodes))

def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs


def get_svd_decomp_stats(U, S, Vh, max_freqs, eigvec_norm='L2'):
    """Compute svd eigen-decomposition-based PE stats of the given graph.

    Args:
        U: Unitary array
        S: Vector(s) with the singular values
        Vh: Unitary array(s)
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(U)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.

    # Normalize and pad eigen vectors.
    if max_freqs < N:
        S = S[:max_freqs]
        U = U[:, :max_freqs]
        Vh = Vh[:max_freqs, :]

        encodings = np.stack((U, Vh.T), axis=-1) * np.expand_dims(np.sqrt(S), axis=-1)
    elif max_freqs > N:
        z = np.zeros((N, max_freqs - N, 2), dtype=np.float32)
        encodings = np.concatenate((np.stack((U, Vh.T), axis=-1) * np.expand_dims(np.sqrt(S), axis=-1), z), axis=1)
    else:
        encodings = np.stack((U, Vh.T), axis=-1) * np.expand_dims(np.sqrt(S), axis=-1)
    return encodings



def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, source, dim=0, dim_size=num_nodes)  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing


def get_heat_kernels_diag(evects, evals, kernel_times=[], space_dim=0):
    """Compute Heat kernel diagonal.

    This is a continuous function that represents a Gaussian in the Euclidean
    space, and is the solution to the diffusion equation.
    The random-walk diagonal should converge to this.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the diffusion diagonal by a factor `t^(space_dim/2)`. In
            euclidean space, this correction means that the height of the
            gaussian stays constant across time, if `space_dim` is the dimension
            of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    heat_kernels_diag = []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels diagonal only for each time
        eigvec_mul = evects ** 2
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j} * phi_{i, j})
            this_kernel = torch.sum(torch.exp(-t * evals) * eigvec_mul,
                                    dim=0, keepdim=False)

            # Multiply by `t` to stabilize the values, since the gaussian height
            # is proportional to `1/t`
            heat_kernels_diag.append(this_kernel * (t ** (space_dim / 2)))
        heat_kernels_diag = torch.stack(heat_kernels_diag, dim=0).transpose(0, 1)

    return heat_kernels_diag


def get_heat_kernels(evects, evals, kernel_times=[]):
    """Compute full Heat diffusion kernels.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
    """
    heat_kernels, rw_landing = [], []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1).unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels for each time
        eigvec_mul = (evects.unsqueeze(2) * evects.unsqueeze(1))  # (phi_{i, j1, ...} * phi_{i, ..., j2})
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j1, ...} * phi_{i, ..., j2})
            heat_kernels.append(
                torch.sum(torch.exp(-t * evals) * eigvec_mul,
                          dim=0, keepdim=False)
            )

        heat_kernels = torch.stack(heat_kernels, dim=0)  # (Num kernel times) x (Num nodes) x (Num nodes)

        # Take the diagonal of each heat kernel,
        # i.e. the landing probability of each of the random walks
        rw_landing = torch.diagonal(heat_kernels, dim1=-2, dim2=-1).transpose(0, 1)  # (Num nodes) x (Num kernel times)

    return heat_kernels, rw_landing


def get_electrostatic_function_encoding(edge_index, num_nodes):
    """Kernel based on the electrostatic interaction between nodes.
    """
    L = to_scipy_sparse_matrix(
        *get_laplacian(edge_index, normalization=None, num_nodes=num_nodes)
    ).todense()
    L = torch.as_tensor(L)
    Dinv = torch.eye(L.shape[0]) * (L.diag() ** -1)
    A = deepcopy(L).abs()
    A.fill_diagonal_(0)
    DinvA = Dinv.matmul(A)

    electrostatic = torch.pinverse(L)
    electrostatic = electrostatic - electrostatic.diag()
    green_encoding = torch.stack([
        electrostatic.min(dim=0)[0],  # Min of Vi -> j
        electrostatic.max(dim=0)[0],  # Max of Vi -> j
        electrostatic.mean(dim=0),  # Mean of Vi -> j
        electrostatic.std(dim=0),  # Std of Vi -> j
        electrostatic.min(dim=1)[0],  # Min of Vj -> i
        electrostatic.max(dim=0)[0],  # Max of Vj -> i
        electrostatic.mean(dim=1),  # Mean of Vj -> i
        electrostatic.std(dim=1),  # Std of Vj -> i
        (DinvA * electrostatic).sum(dim=0),  # Mean of interaction on direct neighbour
        (DinvA * electrostatic).sum(dim=1),  # Mean of interaction from direct neighbour
    ], dim=1)

    return green_encoding


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, HeteroData

class ComputePosencStat(BaseTransform):
    def __init__(self, pe_types, is_undirected, cfg):
        self.pe_types = pe_types
        self.is_undirected = is_undirected
        self.cfg = cfg

    def __call__(self, data: Data) -> Data:
        data = compute_posenc_stats(data, pe_types=self.pe_types,
                                    is_undirected=self.is_undirected,
                                    cfg=self.cfg
                                    )
        return data