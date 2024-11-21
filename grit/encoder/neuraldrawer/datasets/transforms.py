import torch
import torch_geometric as pyg
import numpy as np
import networkx as nx

import torch
import torch_geometric as pyg
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_undirected
import networkx as nx


def pmds_layout(nx_graph):
    import rpy2
    from rpy2.robjects.packages import importr

    igraph = importr('igraph', robject_translations={'.env': '__env'})
    graphlayouts = importr('graphlayouts')
    edge_list = nx.generate_edgelist(nx_graph, data=False)
    edge_list = ' '.join(edge_list).split(' ')
    edge_list = [int(node)+1 for node in edge_list]
    edge_vector = rpy2.robjects.IntVector(edge_list)
    graph = igraph.make_undirected_graph(edges=edge_vector)
    graph = igraph.simplify(graph)
    pos_pivotmds = graphlayouts.layout_with_pmds(graph, pivots=min(10, len(nx_graph.nodes)))
    pos_x = list(pos_pivotmds.rx(True, 1))
    pos_y = list(pos_pivotmds.rx(True, 2))
    pos = {}
    for i in range(len(pos_x)):
        pos[i] = [pos_x[i], pos_y[i]]
    
    return pos

def filter_connected(data):
    G = to_networkx(data, to_undirected=True)
    return G.number_of_nodes() >= 8 and nx.is_connected(G)

def convert_for_stress(data, use_real_edges=True, initializer=nx.drawing.random_layout):
    data.edge_index = pyg.utils.to_undirected(data.edge_index)
    G = to_networkx(data)
    apsp = dict(nx.all_pairs_shortest_path_length(G))
    init_pos = torch.tensor(np.array(list(initializer(G).values())), dtype=torch.float)
    full_edges, attr_d = zip(*[((u, v), d) for u in apsp for v, d in apsp[u].items()])
    full_edge_index, d = pyg.utils.remove_self_loops(*pyg.utils.to_undirected(
        torch.tensor(full_edges).T, torch.tensor(attr_d)
    ))
    start,end = [i for i,j in G.edges],[j for i,j in G.edges]
    edges_actual = torch.tensor([start,end])
    edges_actual = to_undirected(edges_actual)
    k = 1 / d ** 2
    full_edge_attr = torch.stack([d, k], dim=-1)
    if not use_real_edges:
        edges_actual = full_edge_index
    return pyg.data.Data(
        x=init_pos,
        edge_index=edges_actual,
        full_edge_index=full_edge_index,
        full_edge_attr=full_edge_attr,
    )

def convert_for_DeepGD(data, initializer=pmds_layout):
    data.edge_index = pyg.utils.to_undirected(data.edge_index)
    G = to_networkx(data)
    apsp = dict(nx.all_pairs_shortest_path_length(G))
    init_pos = torch.tensor(np.array(list(initializer(G).values())), dtype=torch.float)
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
