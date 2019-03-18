import random
import numpy as np
import scipy
from scipy import sparse
from scipy.stats import norm
import networkx as nx
import community
from generator import ER_generator
import logging

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def to_undirected_graph(G):
    if isinstance(G, nx.classes.graph.Graph):
        return G
    else:
        W = nx.adj_matrix(G)
        W_sym = W + W.T
        G_undirected = nx.Graph(W_sym)
        return G_undirected

def generate_null_model(num_models=10, min_size=40, n=10000, p=0.001, seed=2019, partition=True):
    """Generates a number of null modesl. If partition is True, 
    the graph is partitionned and only one community is chosen randomly.
    """
    models = []
    num_trials = 0
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
            comm_nodes_ = [comm for comm in comm_nodes if len(comm)>=min_size]
            if len(comm_nodes_) == 0:
                if num_trials == 10:
                    comm = max(comm_nodes, key=len)
                else:
                    num_trials += 1
                continue
            random.shuffle(comm_nodes_)
            for comm in comm_nodes_:
                models.append(ER_graph.subgraph(comm))
    return models
        
def break_tie_argsort(l, reverse=False):
    """A re-implementation of np.argsort() addressing tied values:
    when several values are tied, sort them randomly.
    """
    idx_l = list(enumerate(l))
    np.random.shuffle(idx_l)    # to introduce randomness for tied values
    sorted_idx_l = sorted(idx_l, key=lambda x: x[1], reverse=reverse)
    return [idx for idx, v in sorted_idx_l]

def comm_eigenvectors(comm, num_vectors=20):
    W = nx.adjacency_matrix(comm)
    W_sym = W + W.T
    D = np.diag(np.array(W_sym.sum(axis=1)).flatten())
    D = sparse.csr_matrix(D)
    L_comb = D - W_sym
    L_rw = sparse.linalg.inv(D).dot(sparse.csc_matrix(W_sym))
    # eigen vectors
    if W.shape[0] > 2*num_vectors:
        logging.info("Using sparse method to compute eigen vectors")
        _, W_vectors_upper = sparse.linalg.eigs(W_sym, k=num_vectors, which='SM')
        logging.info("Using sparse method to compute eigen vectors")
        _, W_vectors_lower = sparse.linalg.eigs(W_sym, k=num_vectors, which='LM')
    else:
        W_values, W_vectors = scipy.linalg.eig(W_sym.toarray())
        W_sort_index = break_tie_argsort(W_values)
        middle = len(W_values) // 2
        W_vectors_upper = W_vectors[:, W_sort_index[:middle]] # small eigen values
        W_vectors_lower = W_vectors[:, W_sort_index[middle:][::-1]]   # big eigen values
    if W.shape[0] > 21:
        logging.info("Using sparse method to compute eigen vectors")
        _, comb_vectors = sparse.linalg.eigs(L_comb, k=21, which='SM')
        comb_vectors = comb_vectors[:, 1:]
        logging.info("Using sparse method to compute eigen vectors")
        _, rw_vectors = sparse.linalg.eigs(L_rw, k=21, which='LM')
        rw_vectors = rw_vectors[:, 1:]
    else:
        comb_values, comb_vectors = scipy.linalg.eig(L_comb.toarray())
        comb_sort_index = break_tie_argsort(comb_values)
        comb_vectors = comb_vectors[:, comb_sort_index[1:21]]
        rw_values, rw_vectors = scipy.linalg.eig(L_rw.toarray())
        rw_sort_index = break_tie_argsort(rw_values, reverse=True)
        rw_vectors = rw_vectors[:, rw_sort_index[1:21]]
    return W_vectors_upper, W_vectors_lower, comb_vectors, rw_vectors



    

