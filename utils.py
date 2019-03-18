import random
import numpy as np
import networkx as nx
import community

from generator import ER_generator

def to_undirected_graph(G):
    if isinstance(G, nx.classes.graph.Graph):
        return G
    else:
        G_undir  = nx.Graph()
        undir_edges = {}
        for edge_w in G.edges.data('weight', default=0):
            edge = (min(edge_w[0], edge_w[1]), max(edge_w[0], edge_w[1]))
            undir_edges[edge] = undir_edges.get(edge, 0) + edge_w[2]
        G_undir.add_weighted_edges_from([(k[0], k[1], w) for k, w in undir_edges.items()])
        return G_undir

def generate_null_model(num_models=10, min_size=40, n=10000, p=0.001, seed=2019, partition=True):
    """Generates a number of null modesl. If partition is True, 
    the graph is partitionned and only one community is chosen randomly.
    """
    models = []
    while len(models) <= num_models:
        ER_graph = ER_generator(n, p, seed)
        if not partition:
            if nx.number_of_nodes(ER_graph) >= min_size:
                models.append(ER_graph)
        else:
            partition = community.best_partition(to_undirected_graph(ER_graph))
            num_comm = len(set(partition.values()))
            comm_nodes = [[] for _ in range(num_comm)]
            for node, comm_id in partition.items():
                comm_nodes[comm_id].append(node)
            random.shuffle(comm_nodes)
            for comm in comm_nodes:
                if len(comm) >= min_size:
                    models.append(ER_graph.subgraph(comm))
    return models
        




    

