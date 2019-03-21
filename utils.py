import random
import numpy as np
import scipy
from scipy import sparse
from scipy.stats import norm
import networkx as nx
import community
from generator import ER_generator
import logging
from scipy.special import comb
from math import factorial
np.seterr(all='raise')
scipy.seterr(all='raise')

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

# def generate_null_model(num_models=10, min_size=20, n=10000, p=0.001, augment=False):
#     """Generates a number of null modesl. If partition is True, 
#     the graph is partitionned and only one community is chosen randomly.
#     """
#     models = []
#     num_trials = 0
#     while len(models) < num_models:
#         logging.info("Generating {}-th null model".format(len(models)+1))
#         ER_graph = ER_generator(n, p, seed=None)
#         if augment:
#             ER_graph = augmentation(ER_graph)
#         logging.info("Partitioning graph")
#         comm_nodes = partition_graph(ER_graph)
#         comm_nodes_ = [comm for comm in comm_nodes if len(comm)>=2*min_size]
#         if len(comm_nodes_) == 0:
#             logging.warning("No community with enough size, resampling")
#             if num_trials == 10:
#                 logging.warning("Maximum trial reached, take currently biggest community")
#                 comm = max(comm_nodes, key=len)
#                 num_trials = 0
#             else:
#                 num_trials += 1
#         else:
#             num_trials = 0
#             comm = comm_nodes_[np.random.choice(range(len(comm_nodes_)))]
#             models.append(ER_graph.subgraph(comm).copy())
#     return models

def generate_null_models(graph, num_models=10, min_size=20, augment=False):
    """Generates a number of null modesl. If partition is True, 
    the graph is partitionned and only one community is chosen randomly.
    """
    null_models = []
    null_comms = []
    num_trials = 0
    while len(null_models) < num_models:
        logging.info("Generating {}-th null model".format(len(null_models)+1))
        edge_weights = graph.edges.data('weight')
        out_stubs = [e[0] for e in edge_weights]
        in_stubs = [e[1] for e in edge_weights]
        weights = [e[2] for e in edge_weights]
        random.shuffle(in_stubs)
        random.shuffle(weights)
        new_graph = nx.DiGraph()
        new_edge_weights = [(out_node, in_node, w) for out_node, in_node, w in zip(out_stubs, in_stubs, weights) if out_node!=in_node]
        new_graph.add_weighted_edges_from(new_edge_weights)
        if augment:
            new_graph = augmentation(new_graph)
        logging.info("Partitioning graph")
        comm_nodes = partition_graph(new_graph)
        comm_nodes_ = [comm for comm in comm_nodes[1:] if len(comm)>=2*min_size]
        if len(comm_nodes_) == 0:
            if num_trials == 10:
                logging.warning("Maximum trial reached, take currently biggest community")
                comm = max(comm_nodes, key=len)
                num_trials = 0
            else:
                logging.warning("No community with enough size, resampling")
                num_trials += 1
        else:
            num_trials = 0
            comm = comm_nodes_[np.random.choice(range(len(comm_nodes_)))]
        null_comms.append(new_graph.subgraph(comm).copy())
        null_models.append(new_graph)
    return null_models, null_comms
        
def break_tie_argsort(l, reverse=False):
    """A re-implementation of np.argsort() addressing tied values:
    when several values are tied, sort them randomly.
    """
    idx_l = list(enumerate(l))
    np.random.shuffle(idx_l)    # to introduce randomness for tied values
    sorted_idx_l = sorted(idx_l, key=lambda x: x[1], reverse=reverse)
    return [idx for idx, v in sorted_idx_l]

def comm_eigenvectors(comm, num_vectors=20, verbose=False):
    W = nx.adjacency_matrix(comm)
    W_sym = W + W.T
    D_array = np.diag(np.array(W_sym.sum(axis=1)).flatten())
    D = sparse.csc_matrix(D_array)
    L_comb = D - W_sym
    try:
        L_rw = sparse.linalg.inv(D).dot(W_sym)
    except:
        L_rw = sparse.csc_matrix(scipy.linalg.pinv(D_array)).dot(W_sym)
    if verbose:
        W_eig_v, W_vectors = scipy.linalg.eigh(W_sym.toarray())
        W_sort_index = break_tie_argsort(W_eig_v)
        print(W_vectors[:, W_sort_index])
    # eigen vectors
    if W.shape[0] > 2*num_vectors:
        try:
            logging.info("Using sparse method to compute eigen vectors")
            _, W_vectors_upper = sparse.linalg.eigsh(W_sym, k=num_vectors, sigma=0, which='LM')
            logging.info("Using sparse method to compute eigen vectors")
            _, W_vectors_lower = sparse.linalg.eigsh(W_sym, k=num_vectors, which='LM')
        except:
            logging.warning("Sparse method doesn't converge.")
            W_values, W_vectors = scipy.linalg.eigh(W_sym.toarray())
            W_sort_index = break_tie_argsort(W_values)
            W_vectors_upper = W_vectors[:, W_sort_index[:num_vectors]] # small eigen values
            W_vectors_lower = W_vectors[:, W_sort_index[-num_vectors:][::-1]]   # big eigen values
    else:
        W_values, W_vectors = scipy.linalg.eigh(W_sym.toarray())
        W_sort_index = break_tie_argsort(W_values)
        middle = len(W_values) // 2
        W_vectors_upper = W_vectors[:, W_sort_index[:middle]] # small eigen values
        W_vectors_lower = W_vectors[:, W_sort_index[middle:][::-1]]   # big eigen values
    if W.shape[0] > num_vectors+2:
        try:
            logging.info("Using sparse method to compute eigen vectors")
            _, comb_vectors = sparse.linalg.eigsh(L_comb, k=num_vectors+1, sigma=0, which='LM')
            comb_vectors = comb_vectors[:, 1:]
        except:
            logging.warning("Sparse method doesn't converge.")
            comb_values, comb_vectors = scipy.linalg.eigh(L_comb.toarray())
            comb_sort_index = break_tie_argsort(comb_values)
            comb_vectors = comb_vectors[:, comb_sort_index[1:21]]
        try:
            logging.info("Using sparse method to compute eigen vectors")
            _, rw_vectors = sparse.linalg.eigsh(L_rw, k=num_vectors+1, which='LM')
            rw_vectors = rw_vectors[:, 1:]
        except scipy.sparse.linalg.ArpackNoConvergence:
            logging.warning("Sparse method doesn't converge.")
            rw_values, rw_vectors = scipy.linalg.eigh(L_rw.toarray())
            rw_sort_index = break_tie_argsort(rw_values, reverse=True)
            rw_vectors = rw_vectors[:, rw_sort_index[1:21]]
    else:
        comb_values, comb_vectors = scipy.linalg.eigh(L_comb.toarray())
        comb_sort_index = break_tie_argsort(comb_values)
        comb_vectors = comb_vectors[:, comb_sort_index[1:21]]
        rw_values, rw_vectors = scipy.linalg.eigh(L_rw.toarray())
        rw_sort_index = break_tie_argsort(rw_values, reverse=True)
        rw_vectors = rw_vectors[:, rw_sort_index[1:21]]
    return np.real(W_vectors_upper), np.real(W_vectors_lower), np.real(comb_vectors), np.real(rw_vectors)

def verify_clique(p, w, n, k):
    if w * p < 1 - (1 - comb(n, k)**(- 1. / comb(k, 2)))**0.5:
        return True
    return False
def percentile(graph, q=99):
    all_weights = list(nx.get_edge_attributes(graph, 'weight').values())
    return np.percentile(all_weights, q)

def verify_ring(p, w, n, k):
    if w * p < (comb(n, k) * factorial(k-1))**(-1./k):
        return True
    return False
def augmentation(graph):
    g = graph.copy()
    threshold = percentile(graph, q=99)
    n = g.number_of_nodes()
    while True:
        finish = True
        for i in range(n):
            neighbors = g.neighbors(i)
            for neighbor in neighbors:
                w1 = g.get_edge_data(i, neighbor)['weight']
                if w1 > threshold:
                    hop2neighbors = g.neighbors(neighbor)
                    for hop2 in hop2neighbors:
                        w2 = g.get_edge_data(neighbor, hop2)['weight']
                        if hop2 != i and w2 > threshold:
                            if g.get_edge_data(hop2, i) is None:
                                g.add_edge(hop2, i, weight=min([w1, w2]))
                                finish = False
                            else:
                                w3 = g.get_edge_data(hop2, i)['weight']
                                if w3 < w1 and w3 < w2:
                                    g[hop2][i]['weight'] = min(w1, w2)
                                    finish = False
        if finish:
            break
    return g

def verify_path(p, w, n, k):
    if w * p < (comb(n, k) * factorial(k))**(-1./(k-1)):
        return True
    return False
# def precicion_recall(pred, real, *sample_sizes):
#     pred = np.array(pred)
#     real = np.array(real)
#     right_anomaly = (real==1) & (pred==1)
#     # for 
#     num_anormaly_samples = np.sum(right_anomaly[:sample_size])
#     prec = num_anormaly_samples / sample_size
#     rec = num_anormaly_samples / np.sum(real)
#     return pred, 

def verify_star(p, w, n, k):
    for k1 in range(k):
        if not w * p < (comb(n, k) * comb(k, k1) * (k-k1)) ** (-1./(k-1)):
            return False
    return True

def verify_tree(p, w, n):
    if w * p < (4 * comb(n, 9) * comb(9, 5)) ** (-1./18):
        return True
    return False

def get_parameters(n, ps, ws):
    res = []
    for p in ps:
        for w in ws:
            add = True
            for k in range(5, 21):
                if not verify_clique(p, w, n, k) or not verify_ring(p, w, n, k) or not verify_path(p, w, n, k) or not verify_star(p, w, n, k) or not verify_tree(p, w, n):
                    add = False
                    break
            if add:
                res.append((p, w))
    return res

ps = np.linspace(0.01, 0.05, num=5)
ws = np.linspace(0.0, 0.01, num=11)

res = get_parameters(2000, ps, ws)
print(res)


