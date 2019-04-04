'''
This file includes the implementation of NetEMD module.
'''

from rpy2.robjects import FloatVector
import rpy2.robjects.packages as rpackages
netdist = rpackages.importr('netdist')

import numpy as np
import community
import networkx as nx
from utils import generate_null_models, comm_eigenvectors, partition_graph
from scipy.stats import norm
from com_detection import augmentation
from generator import ER_generator
import logging
np.seterr(all='raise')

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

def NetEMD_to_ref(stat, ref_stats):
    dist = [compute_NetEMD(stat, ref.values()) for ref in ref_stats]
    dist.remove(min(dist))
    dist.remove(max(dist))
    return np.mean(dist)

def compute_NetEMD(u1, u2):
    u1 = np.array(list(u1))
    u1 = u1 * (u1 > 1e-12)
    h1 = np.histogram(u1, bins='auto', density=False)
    h1_loc = (h1[1][:-1] + h1[1][1:]) / 2
    u2 = np.array(list(u2))
    u2 = u2 * (u2 > 1e-12)
    h2 = np.histogram(u2, bins='auto', density=False)
    h2_loc = (h2[1][:-1] + h2[1][1:]) / 2
    dhist1 = netdist.dhist(FloatVector(h1_loc), FloatVector(h1[0]))
    dhist2 = netdist.dhist(FloatVector(h2_loc), FloatVector(h2[0]))
    dist = netdist.net_emd(dhist1, dhist2)[0]
    return dist

def NetEMD_score(obs_stat, ref_stats, null_stats):
    # dicts
    # ref_stats, null_stats already normalized
    # return dictionary
    obs_values = np.array(list(obs_stat.values()))
    obs_mean = np.mean(obs_values)
    try:
        assert isinstance(obs_mean, float)
    except:
        print("assert obs_mean float failed")
        print(obs_mean)
        raise Exception("error caught!")
    obs_std = np.std(obs_values)
    obs_normed = obs_values / obs_std if obs_std!=0 else obs_values
    dist_obs_ref = NetEMD_to_ref(obs_normed, ref_stats)
    dist_null_ref = np.array([NetEMD_to_ref(null_stat, ref_stats) for null_stat in null_stats])
    p_value = (np.sum(dist_null_ref>dist_obs_ref) + 1) / len(null_stats)
    nodes = list(obs_stat.keys())
    score_1 = {}
    score_2 = {}
    if p_value > 0.05:
        score_1 = {node: 0 for node in nodes}
        score_2 = {node: 0 for node in nodes}
    else:
        scaled_stat = (obs_values-obs_mean) / obs_std if obs_std!= 0 else obs_values-obs_mean
        scaled_stat = np.abs(scaled_stat)
        p_score = norm.ppf(1-p_value)
        p_score = np.clip(p_score, a_min=-8, a_max=8)
        top5_args = np.argsort(scaled_stat)[::-1][:int(np.ceil(len(nodes)*0.05))]
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
        std = np.std(values)
        normed = values / std if std!=0 else values
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
    motif_id = motif_dict[tuple(res)]
    edge_weights = [e_w[2] for e_w in g.edges(data='weight', default=1)]
    return motif_id, np.prod(edge_weights)

def compute_motif_stat(g, normalize=False):
    """Returns an array of size: num_nodes, 13
    """
    nodes = list(g.nodes())
    stats = np.zeros((len(nodes), 13), dtype=float)
    for i, node in enumerate(nodes):
        hop1 = set(g.successors(node)) | set(g.predecessors(node))
        for a in hop1:
            hop2 = set(g.successors(a)) | set(g.predecessors(a)) | hop1
            for b in hop2:
                if a < b and b != node:
                    tri_sub_g = g.subgraph([node, a, b])
                    motif_id, edge_prod = motif_id_and_weight_prod(tri_sub_g)
                    stats[i][motif_id-4] += edge_prod
    if normalize:
        std = np.std(stats, axis=0)
        non_zero_idx = np.where(std!=0)[0]
        stats[:, non_zero_idx] /= std[non_zero_idx]
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
    direction = 2 * (num_pos >= num_neg) - 1
    vectors *= direction
    equal_cols = np.where(num_pos==num_neg)[0]
    logging.debug("found {} columns with equal num_pos and num_neg".format(len(equal_cols)))
    for eq_col_idx, eq_col in enumerate(equal_cols):
        logging.debug("treating equal col {}/{}".format(eq_col_idx+1, len(equal_cols)))
        v = vectors[:, eq_col].copy()
        if all(v==0): continue
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
    logging.debug("generating eigen vectors from graph")
    matrices = comm_eigenvectors(g, num_vectors=5)
    nodes = list(g.nodes())
    matrix_stats = [[] for _ in range(4)]
    for idx, matrix in enumerate(matrices):
        logging.debug("computing matrix stats for matrix type.{}".format(idx))
        vectors_directed = direction_eigenvectors(matrix)
        if normalize:
            std = np.std(vectors_directed, axis=0)
            non_zero_idx = np.where(std!=0)[0]
            if len(np.where(std==0)[0]) != 0:
                print("number of nodes", len(nodes))
                print(vectors_directed)
            vectors_directed[:, non_zero_idx] /= std[non_zero_idx]
        for col in range(min(5, vectors_directed.shape[1])):
            matrix_stat = {node: stat for node, stat in zip(nodes, vectors_directed[:, col])}
            matrix_stats[idx].append(matrix_stat)
    return matrix_stats

def compute_matrix_score(g, ref_stats, null_stats):
    nodes = g.nodes()
    obs_stat = compute_matrix_stat(g) # 4 * 5 dicts

    matrix_scores = []
    for i in range(4):
        matrix_score = np.zeros((2, nx.number_of_nodes(g)), dtype=float)
        for j in range(5):
            obs_stat_ij = obs_stat[i][j]
            ref_stats_ij = [ref[i][j] for ref in ref_stats]
            null_stats_ij = [n_samp[i][j] for n_samp in null_stats]
            try:
                matrix_score_ij_1, matrix_score_ij_2 = NetEMD_score(obs_stat_ij, ref_stats_ij, null_stats_ij)
                matrix_score += np.array([list(matrix_score_ij_1.values()), list(matrix_score_ij_2.values())])
            except MemoryError:
                logging.warning("Memory Error")
        matrix_scores.append((dict(zip(nodes, matrix_score[0])), dict(zip(nodes, matrix_score[1]))))
    return matrix_scores

def NetEMD_features(graph, references, null_samples, num_references=15, num_samples=500):
    logging.info("partition graph")
    communities = [graph.subgraph(comm_nodes) for comm_nodes in partition_graph(graph) if len(comm_nodes) > 4]
    logging.info("got {} communities".format(len(communities)))
    assert len(references) >= num_references
    references = references[:num_references]
    assert len(null_samples) >= num_samples
    null_samples = null_samples[:num_samples]
    for comm_idx, community in enumerate(communities):
        logging.info("computing strength scores for community No.{}/{}".format(comm_idx, len(communities)))
        strength_scores = compute_strength_score(community, references, null_samples)
        strength_names = ['in_strength_1', 'in_strength_2', 'out_strength_1', 'out_strength_2', 'in_out_strength_1', 'in_out_strength_2']
        for strength_name, strength_score in zip(strength_names, strength_scores):
            for node, score in strength_score.items():
                assert graph.node[node].get(strength_name) is None
                graph.node[node][strength_name] = score
        logging.info("computing motif scores for community No.{}/{}".format(comm_idx, len(communities)))
        motif_scores = compute_motif_score(community, references, null_samples)
        for idx in range(13):
            motif_score = motif_scores[idx]
            motif_id = idx + 4
            for score_idx in [1, 2]:
                for node, score in motif_score[score_idx-1].items():
                    assert graph.node[node].get('motif_{}_{}'.format(motif_id, score_idx)) is None
                    graph.node[node]['motif_{}_{}'.format(motif_id, score_idx)] = score
    logging.info("generating augmented graph")
    graph_aug = augmentation(graph)
    logging.info("partition augmented graph")
    communities_aug = [graph_aug.subgraph(comm_nodes) for comm_nodes in partition_graph(graph_aug) if len(comm_nodes)>4]
    logging.info("get {} augmented communities".format(len(communities_aug)))
    ref_stats = []
    for ref_idx, ref in enumerate(references):
        if len(ref.nodes()) < 10:
            print(len(ref.nodes()))
            raise AssertionError
        logging.info("computing matrix stats for refrence No.{}".format(ref_idx))
        ref_stats.append(compute_matrix_stat(ref, normalize=True))
    null_stats = []
    for null_idx, n_samp in enumerate(null_samples):
        if len(n_samp.nodes()) < 10:
            print(len(n_samp.nodes))
            raise AssertionError
        logging.info("computing matrix stats for null_sample No.{}".format(null_idx))
        null_stats.append(compute_matrix_stat(n_samp, normalize=True))
    matrix_names = ['upper', 'lower', 'comb', 'rw']
    for comm_idx, community in enumerate(communities_aug):
        logging.info("computing matrix scores for community No.{}/{}".format(comm_idx, len(communities)))
        matrix_scores = compute_matrix_score(community, ref_stats, null_stats) # 4 tuples of (score1, score2)
        for matrix_idx, matrix_name in enumerate(matrix_names):
            for node in community.nodes():
                graph.node[node]['NetEMD_{}_1'.format(matrix_name)] = matrix_scores[matrix_idx][0][node]
                graph.node[node]['NetEMD_{}_2'.format(matrix_name)] = matrix_scores[matrix_idx][1][node]
    return graph

# from generator import ER_generator, draw_anomalies
# graph = ER_generator(n=500, p=0.01, seed=None)
# graph = draw_anomalies(graph)

# _, references = generate_null_models(graph, num_models=3, min_size=5)
# _, null_samples = generate_null_models(graph, num_models=5, min_size=5)
# graph = NetEMD_features(graph, references, null_samples, num_references=3, num_samples=5)


# print("FINISH!")


