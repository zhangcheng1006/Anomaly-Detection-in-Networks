import rpy2
print(rpy2.__version__)

import rpy2.situation
for row in rpy2.situation.iter_info():
    print(row)

# import rpy2's package module
import rpy2.robjects.packages as rpackages

# import R's utility package
# utils = rpackages.importr('utils')

# select a mirror for R packages
# utils.chooseCRANmirror(ind=1) # select the first mirror in the list

# # R package names
# packnames = ('ggplot2', 'devtools', 'netdist')

# # R vector of strings
# from rpy2.robjects.vectors import StrVector

# # Selectively install what needs to be install.
# # We are fancy, just because we can.
# names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
# print(names_to_install)
# if len(names_to_install) > 0:
#     utils.install_packages(StrVector(names_to_install))

import rpy2.robjects as robjects

netdist = rpackages.importr('netdist')

import numpy as np
import community
import networkx as nx
from utils import to_undirected_graph, generate_null_model, comm_eigenvectors
from com_detection import get_partition
from scipy.stats import norm
from com_detection import augmentation

motif_dict = {  ((0, 2), (1, 0), (1, 0)): 4, 
                ((0, 1), (0, 1), (2, 0)): 5, 
                ((0, 1), (1, 0), (1, 1)): 6, 
                ((0, 1), (1, 1), (2, 1)): 7, 
                ((1, 0), (1, 1), (1, 2)): 8, 
                ((1, 1), (1, 1), (2, 2)): 9, 
                ((0, 2), (1, 1), (2, 0)): 10, 
                ((1, 1), (1, 1), (1, 1)): 11, 
                ((0, 2), (2, 1), (2, 1)): 12, 
                ((1, 2), (1, 2), (2, 0)): 13, 
                ((1, 1), (1, 2), (2, 1)): 14, 
                ((1, 2), (2, 1), (2, 2)): 15, 
                ((2, 2), (2, 2), (2, 2)): 16}

stats = ['in_strength', 'out_strength', 'in_out_strength', 'motif_4_stat', 'motif_5_stat', 'motif_6_stat', 
         'motif_7_stat', 'motif_8_stat', 'motif_9_stat', 'motif_10_stat', 'motif_11_stat', 'motif_12_stat', 
         'motif_13_stat', 'motif_14_stat', 'motif_15_stat', 'motif_16_stat', 'upper_1_netemd', 'upper_2_netemd', 
         'upper_3_netemd', 'upper_4_netemd', 'upper_5_netemd', 'lower_1_netemd', 'lower_2_netemd', 'lower_3_netemd', 
         'lower_4_netemd', 'lower_5_netemd', 'comb_1_netemd', 'comb_2_netemd', 'comb_3_netemd', 'comb_4_netemd', 
         'comb_5_netemd', 'rw_1_netemd', 'rw_2_netemd', 'rw_3_netemd', 'rw_4_netemd', 'rw_5_netemd']

def strength_stats(graph, communities):
    for sub_g in communities:
        for node in sub_g.nodes():
            graph.node[node]['in_strength'] = sum([data[2] for data in sub_g.in_edges(node, data='weight')])
            graph.node[node]['out_strength'] = sum([data[2] for data in sub_g.out_edges(node, data='weight')])
            graph.node[node]['in_out_strength'] = graph.node[node]['in_strength'] + graph.node[node]['out_strength']
    return graph

def get_motif(g):
    global motif_dict
    in_degrees = [in_degree for node, in_degree in g.in_degree()]
    out_degrees = [out_degree for node, out_degree in g.out_degree()]
    res = sorted(zip(in_degrees, out_degrees), key=lambda x: (x[0], x[1]))
    motif = motif_dict.get(tuple(res))
    return motif

def motif_stats(graph, communities, num_motif=13):
    for sub_g in communities:
        all_motif_stats = np.zeros(num_motif)
        for node in sub_g.nodes():
            hop1 = set(sub_g.in_edges(node, data='weight'))|set(sub_g.out_edges(node, data='weight'))
            for _, a in hop1:
                hop2 = set(sub_g.in_edges(a, data='weight'))|set(sub_g.out_edges(a, data='weight'))|hop1
                for _, b in hop2:
                    if a < b:
                        tri_sub_g = sub_g.subgraph([node, a, b])
                        motif = get_motif(tri_sub_g)
                        stat = np.prod([data[2] for data in tri_sub_g.edges.data('weight', default=1)])
                        all_motif_stats[motif-4] += stat
            for i in range(num_motif):
                graph.node[node]['motif_'+str(i+4)+'_stat'] = all_motif_stats[i]
    return graph

def compute_hist(g, stat):
    T_stats = [data[1] for data in g.nodes.data(stat, default=0)]
    return np.histogram(T_stats, bins='auto', density=True)

def compute_NetEMD(g1, g2, stat):
    h1 = compute_hist(g1, stat)
    h2 = compute_hist(g2, stat)
    dhist1 = netdist.dhist(h1[1], h1[0])
    dhist2 = netdist.dhist(h2[1], h2[0])
    dist = netdist.net_emd(dhist1, dhist2)
    return dist

def assign_NetEMD_score(graph, num_references=15, num_samples=500):
    global stats
    communities = get_partition(graph).values()
    num_communities = len(communities)
    references = generate_null_model(num_models=num_references, min_size=5)
    samples = generate_null_model(num_models=num_samples, min_size=5)
    graph = strength_stats(graph, communities)
    graph_aug = augmentation(graph)
    communities_aug = get_partition(graph_aug).values()
    graph = motif_stats(graph, communities)
    graph = matrix_stats(graph, communities)
    for idx, _ in enumerate(references):
        references[idx] = strength_stats(references[idx], [references[idx]])
        references[idx] = motif_stats(references[idx], [references[idx]])
    for idx, _ in enumerate(samples):
        samples[idx] = strength_stats(samples[idx], [samples[idx]])
        samples[idx] = strength_stats(samples[idx], [samples[idx]])


    graph_aug = augmentation(graph)
    NetEMD_matrix_stats = matrix_stats(graph, references)
    # TODO: if reference augmented, resample


    y_M = np.zeros((num_samples, 36))
    for i in range(36):
        for j, sam_g in enumerate(samples):
            dists = []
            for ref_g in references:
                dists.append(compute_NetEMD(sam_g, ref_g, stats[i]))
            dists.remove(min(dists))
            dists.remove(max(dists))
            y_M[j, i] = np.mean(dists)

    y = np.zeros((num_communities, 36))
    for i in range(36):
        for j, sub_g in enumerate(communities):
            dists = []
            for ref_g in references:
                dists.append(compute_NetEMD(sub_g, ref_g, stats[i]))
            dists.remove(min(dists))
            dists.remove(max(dists))
            y[j, i] = np.mean(dists)

            p_value = (sum(y_M[:, i] > y[j, i])+1) / num_samples
            if p_value >= 0.05:
                for node in sub_g.nodes():
                    graph.node[node]['NetEMD_score_1_'+str(i+1)] = 0
                    graph.node[node]['NetEMD_score_2_'+str(i+1)] = 0
            else:
                T_stats = np.reshape([[data[0], data[1]] for data in sub_g.nodes.data(stats[i], default=0)], (-1, 2))
                mean = np.mean(T_stats[:, 1])
                std = np.std(T_stats[:, 1])
                for k in range(T_stats.shape[0]):
                    node = T_stats[k, 0]
                    value = T_stats[k, 1]
                    criterion = abs(value - mean) / std
                    if criterion < 2:
                        graph.node[node]['NetEMD_score_1_'+str(i+1)] = 0
                    else:
                        graph.node[node]['NetEMD_score_1_'+str(i+1)] = criterion
                indexes = np.argsort(T_stats[:, 1])
                bottom = int(T_stats.shape[0] * 0.95) + 1
                for idx in range(bottom):
                    node = T_stats[indexes[idx], 0]
                    graph.node[node]['NetEMD_score_2_'+str(i+1)] = 0
                for idx in range(bottom, T_stats.shape[0]):
                    node = T_stats[indexes[idx], 0]
                    graph.node[node]['NetEMD_score_2_'+str(i+1)] = norm.ppf(1 - p_value)

    for i in range(4):
        for node in graph.nodes():
            value1 = 0
            value2 = 0
            for j in range(5):
                value1 += graph.node[node]['NetEMD_score_1_'+str(17 + i*5 + j)]
                value2 += graph.node[node]['NetEMD_score_2_'+str(17 + i*5 + j)]
            graph.node[node]['Compact_NetEMD_score_1_'+str(i+1)] = value1
            graph.node[node]['Compact_NetEMD_score_2_'+str(i+1)] = value2

    return graph

def matrix_stats(graph, communities):
    for sub_g in communities:
        nodes = sub_g.nodes()
        matrices = comm_eigenvectors(sub_g, num_vectors=5)
        matrices_name = ['upper', 'lower', 'comb', 'rw']
        for matrix_name, vectors in zip(matrices_name, matrices):
            vectors_dir_norm = direction_eigenvectors(vectors)
            for col in range(5):
                for node_id, node in enumerate(nodes):
                    graph.node[node]['{}_{}_netemd'.format(matrix_name, col+1)] = vectors_dir_norm[node_id, col]
    return graph



def direction_eigenvectors(vectors_origine):
    """Normalize the direction of the eigenvectors according to Appendix.D
    vectors is a 2D matrix, each column an eigen vector
    """
    vectors = vectors_origine.copy()
    num_pos = np.sum(vectors>0, axis=0)
    num_neg = np.sum(vectors<0, axis=0)
    direction = num_pos >= num_neg
    vectors *= direction
    equal_cols = np.where(num_pos == num_neg)
    for eq_col in equal_cols:
        v = vectors[:, eq_col].copy()
        v_2 = np.power(v, 2)
        while True:
            if sum(v) < 0:
                vectors[:, eq_col] *= -1
                break
            elif sum(v) > 0:
                break
            v *= v_2
    return vectors
