import rpy2
print(rpy2.__version__)

import rpy2.situation
for row in rpy2.situation.iter_info():
    print(row)

# import rpy2's package module
import rpy2.robjects.packages as rpackages

# import R's utility package
utils_R = rpackages.importr('utils')

# select a mirror for R packages
utils_R.chooseCRANmirror(ind=1) # select the first mirror in the list

# # R package names
# packnames = ('ggplot2', 'devtools', 'netdist')

# # R vector of strings
# from rpy2.robjects.vectors import StrVector

# # Selectively install what needs to be install.
# # We are fancy, just because we can.
# names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
# print(names_to_install)
# if len(names_to_install) > 0:
#     utils_R.install_packages(StrVector(names_to_install))

netdist = rpackages.importr('netdist')

import numpy as np
import community
import networkx as nx
from utils import generate_null_model, comm_eigenvectors, partition_graph
from scipy.stats import norm
from com_detection import augmentation
from generator import *
import logging

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

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
# compute NetEMD
# compute Hist

def NetEMD_to_ref(stat, ref_stats):
    dist = [compute_NetEMD(stat, ref.values()) for ref in ref_stats]
    dist.remove(min(dist))
    dist.remove(max(dist))
    return np.mean(dist)

def compute_NetEMD(u1, u2):
    h1 = np.histogram(list(u1), bins='auto', density=True)
    h2 = np.histogram(list(u2), bins='auto', density=True)
    dhist1 = netdist.dhist(h1[1], h1[0])
    dhist2 = netdist.dhist(h2[1], h2[0])
    dist = netdist.net_emd(dhist1, dhist2)
    return dist

def NetEMD_score(obs_stat, ref_stats, null_stats):
    # dicts
    # ref_stats, null_stats already normalized
    # return dictionary
    obs_values = np.array(list(obs_stat.values()))
    obs_mean = np.mean(obs_values)
    assert isinstance(obs_mean, float)
    obs_std = np.std(obs_values)
    obs_normed = obs_values / obs_std
    dist_obs_ref = NetEMD_to_ref(obs_normed, ref_stats)
    dist_null_ref = np.array([NetEMD_to_ref(null_stat, ref_stats) for null_stat in null_stats])
    p_value = (np.sum(dist_null_ref>dist_obs_ref) + 1) / len(null_stats)
    score_1 = {}
    score_2 = {}
    scaled_stat = np.abs((obs_values-obs_mean) / obs_std)
    nodes = list(obs_stat.keys())
    for nid, node in enumerate(nodes):
        score_1[node] = 0 if scaled_stat[nid]<2 else scaled_stat[nid]
    p_score = norm.ppf(1-p_value)
    top5_args = np.argsort(scaled_stat)[::-1][:np.ceil(len(obs_stat)*0.05)]
    for nid, node in enumerate(nodes):
        score_1[node] = 0 if scaled_stat[nid]<2 else scaled_stat[nid]
        score_2[node] = p_score if nid in top5_args else 0
    return score_1, score_2

def compute_strength(g, strength_type=None, normalize=False):
    if strength_type == 'in':
        strength = dict(g.in_degree(weight='weight'))
    elif strength_type == 'out':
        strength = dict(g.out_degree(weight='weight'))
    else:
        strength = dict(g.degree(weight='weight'))
    if normalize:
        values = np.array(list(strength.values()))
        normed = values / np.std(values)
        return {node: v for node, v in zip(strength.keys(), normed)}
    return strength

def compute_strength_score(g, references, null_samples):
    """Compute one statistics on a graph(subgraph)
    """
    # in strength
    obs_stat = compute_strength(g, 'in')
    ref_stats = [compute_strength(ref, 'in', normalize=True) for ref in references]
    null_stats = [compute_strength(n_samp, 'in', normalize=True) for n_samp in null_samples]
    in_strength_1, in_strength_2 = NetEMD_score(obs_stat, ref_stats, null_stats)
    # out_strength
    obs_stat = compute_strength(g, 'out')
    ref_stats = [compute_strength(ref, 'out', normalize=True) for ref in references]
    null_stats = [compute_strength(n_samp, 'out', normalize=True) for n_samp in null_samples]
    out_strength_1, out_strength_2 = NetEMD_score(obs_stat, ref_stats, null_stats)
    # in_out_strength
    obs_stat = compute_strength(g)
    ref_stats = [compute_strength(ref, normalize=True) for ref in references]
    null_stats = [compute_strength(n_samp, normalize=True) for n_samp in null_samples]
    in_out_strength_1, in_out_strength_2 = NetEMD_score(obs_stat, ref_stats, null_stats)
    
    return in_strength_1, in_strength_2, out_strength_1, out_strength_2, in_out_strength_1, in_out_strength_2

def motif_id_and_weight_prod(g):
    global motif_dict
    in_degrees = [in_degree for node, in_degree in g.in_degree()]
    out_degrees = [out_degree for node, out_degree in g.out_degree()]
    res = sorted(zip(in_degrees, out_degrees), key=lambda x: (x[0], x[1]))
    motif_id = motif_dict.get(tuple(res))
    edge_weights = [e_w[2] for e_w in g.edges(data='weight', default=1)]
    return motif_id, np.prod(edge_weights)

def compute_motif_stat(g, normalize=False):
    """Returns an array of size: num_nodes, 13
    """
    nodes = list(g.nodes())
    stats = np.zeros((len(nodes), 13))
    for i, node in enumerate(nodes):
        hop1 = set(g.successors(node)) | set(g.predecessors(node))
        for a in hop1:
            hop2 = set(g.successors(a)) | set(g.predecessors(a)) | hop1
            for b in hop2:
                if a < b:
                    tri_sub_g = g.subgraph([node, a, b])
                    motif_id, edge_prod = motif_id_and_weight_prod(tri_sub_g)
                    stats[i][motif_id-4] += edge_prod
    if normalize:
        std = np.std(stats, axis=0)
        stats /= std
    for i in range(13):
        # yield a dict
        yield {node: motif_stat for node, motif_stat in zip(nodes, stats[:, i])}

def compute_motif_score(g, references, null_samples):
    obs_stat = compute_motif_stat(g)
    ref_stats = [compute_motif_stat(ref, normalize=True) for ref in references] # list of generators
    null_stats = [compute_motif_stat(n_samp, normalize=True) for n_samp in null_samples]
    motif_scores = []
    for _ in range(13):
        obs_stat_i = next(obs_stat)
        ref_stats_i = [next(ref) for ref in ref_stats] # list of dicts
        null_stats_i = [next(n_samp) for n_samp in null_stats]
        motif_score = NetEMD_score(obs_stat_i, ref_stats_i, null_stats_i)
        motif_scores.append(motif_score)
    return motif_scores

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

def compute_matrix_stat(g, normalize=False):
    """returns a list of dictionaries
    """
    matrices = comm_eigenvectors(g, num_vectors=5)
    nodes = list(g.nodes())
    matrix_stats = [[] for _ in range(4)]
    for idx, matrix in enumerate(matrices):
        vectors_directed = direction_eigenvectors(matrix)
        if normalize:
            vectors_directed /= np.std(vectors_directed, axis=0)
        for col in range(5):
            matrix_stat = {node: stat for node, stat in zip(nodes, vectors_directed[:, col])}
            matrix_stats[idx].append(matrix_stat)
    return matrix_stats

def compute_matrix_score(g, references, null_samples):
    nodes = g.nodes()
    obs_stat = compute_matrix_stat(g) # 4 * 5 dicts
    ref_stats = [compute_matrix_stat(ref, normalize=True) for ref in references]
    null_stats = [compute_matrix_stat(n_samp, normalize=True) for n_samp in null_samples]
    matrix_scores = []
    for i in range(4):
        matrix_score = np.zeros((2, nx.number_of_nodes(g)), dtype=float)
        for j in range(5):
            obs_stat_ij = obs_stat[i][j]
            ref_stats_ij = [ref[i][j] for ref in ref_stats]
            null_stats_ij = [n_samp[i][j] for n_samp in null_stats]
            matrix_score_ij_1, matrix_score_ij_2 = NetEMD_score(obs_stat_ij, ref_stats_ij, null_stats_ij)
            matrix_score += np.array([list(matrix_score_ij_1.values()), list(matrix_score_ij_2.values())])
        matrix_scores.append((dict(zip(nodes, matrix_score[0])), dict(zip(nodes, matrix_score[1]))))
    return matrix_scores

def NetEMD_features(graph, num_references=15, num_samples=500, n=10000, p=0.001):
    global stats
    logging.info("partition graph")
    communities = [graph.subgraph(comm_nodes) for comm_nodes in partition_graph(graph)]
    logging.info("got {} communities".format(len(communities)))
    logging.info("generating references")
    references = generate_null_model(num_models=num_references, min_size=5, n=n, p=p)
    logging.info("generating null samples")
    null_samples = generate_null_model(num_models=num_samples, min_size=5, n=n, p=p)
    for comm_idx, community in enumerate(communities):
        logging.info("computing strength scores for community No.{}".format(comm_idx))
        strength_scores = compute_strength_score(community, references, null_samples)
        strength_names = ['in_strength_1', 'in_strength_2', 'out_strength_1', 'out_strength_2', 'in_out_strength_1', 'in_out_strength_2']
        for strength_name, strength_score in zip(strength_names, strength_scores):
            for node, score in strength_score.items():
                assert graph[node].get(strength_name) is None
                graph[node][strength_name] = score
        logging.info("computing motif scores for community No.{}".format(comm_idx))
        motif_scores = compute_motif_score(community, references, null_samples)
        for idx in range(13):
            motif_score = motif_scores[idx]
            motif_id = idx + 4
            for score_idx in [1, 2]:
                for node, score in motif_score[score_idx-1].items():
                    assert graph[node].get('motif_{}_{}'.format(motif_id, score_idx)) is None
                    graph[node]['motif_{}_{}'.format(motif_id, score_idx)] = score
    logging.info("generating augmented graph")
    graph_aug = augmentation(graph)
    logging.info("partition augmented graph")
    communities_aug = [graph_aug.subgraph(comm_nodes) for comm_nodes in partition_graph(graph_aug)]
    logging.info("get {} augmented communities".format(len(communities_aug)))
    logging.info("generating augmented refrences")
    references_aug = generate_null_model(num_models=num_references, min_size=5, n=n, p=p, augment=True)
    logging.info("generating augmented null samples")
    null_samples_aug = generate_null_model(num_models=num_samples, min_size=5, n=n, p=p, augment=True)
    matrix_names = ['upper', 'lower', 'comb', 'rw']
    for comm_idx, community in enumerate(communities_aug):
        logging.info("computing matrix scores for community No.{}".format(comm_idx))
        matrix_scores = compute_matrix_score(community, references_aug, null_samples_aug) # 4 tuples of (score1, score2)
        for matrix_idx, matrix_name in enumerate(matrix_names):
            for node in community.nodes():
                graph[node]['{}_1'.format(matrix_name)] = matrix_scores[matrix_idx][0][node]
                graph[node]['{}_2'.format(matrix_name)] = matrix_scores[matrix_idx][1][node]
    
    return graph

graph = ER_generator(n=500, p=0.005, seed=None)
graph = draw_anomalies(graph)
graph = NetEMD_features(graph, num_references=2, num_samples=2, n=500)

