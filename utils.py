import random
import numpy as np
import scipy
from scipy import sparse
from scipy.stats import norm
import networkx as nx
import community
from generator import ER_generator
from com_detection import augmentation
import logging

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def to_undirected_graph(G):
    if isinstance(G, nx.classes.digraph.DiGraph):
        W = nx.adj_matrix(G)
        W_sym = W + W.T
        G_undirected = nx.Graph(W_sym)
        return G_undirected
    else:
        return G

def partition_graph(G):
    """Partition a graph and return the nodes ressembled by community
    """
    Undir_graph = to_undirected_graph(G)
    partitions = community.best_partition(Undir_graph)
    num_comm = len(set(partitions.values()))
    comm_nodes = [[] for _ in range(num_comm)]
    for node, comm_id in partitions.items():
        comm_nodes[comm_id].append(node)
    return comm_nodes

def generate_null_model(num_models=10, min_size=40, n=10000, p=0.001, augment=False):
    """Generates a number of null modesl. If partition is True, 
    the graph is partitionned and only one community is chosen randomly.
    """
    models = []
    num_trials = 0
    while len(models) < num_models:
        logging.info("Generating {}-th null model".format(len(models)+1))
        ER_graph = ER_generator(n, p, seed=None)
        if augment:
            ER_graph = augmentation(ER_graph)
        logging.info("Partitioning graph")
        comm_nodes = partition_graph(ER_graph)
        comm_nodes_ = [comm for comm in comm_nodes if len(comm)>=min_size]
        if len(comm_nodes_) == 0:
            logging.warning("No community with enough size, resampling")
            if num_trials == 10:
                logging.warning("Maximum trial reached, take currently biggest community")
                comm = max(comm_nodes, key=len)
                num_trials = 0
            else:
                num_trials += 1
        else:
            num_trials = 0
            comm = comm_nodes_[np.random.choice(range(len(comm_nodes_)))]
            models.append(ER_graph.subgraph(comm).copy())
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
    if W.shape[0] > num_vectors+2:
        logging.info("Using sparse method to compute eigen vectors")
        _, comb_vectors = sparse.linalg.eigs(L_comb, k=num_vectors+1, which='SM')
        comb_vectors = comb_vectors[:, 1:]
        logging.info("Using sparse method to compute eigen vectors")
        _, rw_vectors = sparse.linalg.eigs(L_rw, k=num_vectors+1, which='LM')
        rw_vectors = rw_vectors[:, 1:]
    else:
        comb_values, comb_vectors = scipy.linalg.eig(L_comb.toarray())
        comb_sort_index = break_tie_argsort(comb_values)
        comb_vectors = comb_vectors[:, comb_sort_index[1:21]]
        rw_values, rw_vectors = scipy.linalg.eig(L_rw.toarray())
        rw_sort_index = break_tie_argsort(rw_values, reverse=True)
        rw_vectors = rw_vectors[:, rw_sort_index[1:21]]
    return W_vectors_upper, W_vectors_lower, comb_vectors, rw_vectors


    

