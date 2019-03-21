import numpy as np
import scipy
from scipy import sparse
from scipy.stats import norm
import networkx as nx
import logging
from utils import comm_eigenvectors, partition_graph, generate_null_models

np.seterr(all='raise')

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def null_norm_based_params(null_vectors):
    """Gets useful statistic parameters on the null eigenvectors.
    Parameters:
    -----------
        null_vectors: matrix of size: [m, n, 20]
    """
    null_max = np.zeros((len(null_vectors), 20), dtype=float)
    for m, null_vs in enumerate(null_vectors):
        assert null_vs.shape[1] == 20
        null_abs = np.abs(null_vs)
        null_max[m, :] = np.max(null_abs, axis=0)
    null_mean = np.mean(null_max, axis=0)
    null_std = np.std(null_max, axis=0)
    return null_max, null_mean, null_std

def norm_based_localization_1(obs_vectors):
    num_nodes = obs_vectors.shape[0]
    obs_abs = np.abs(obs_vectors)
    g1 = obs_abs * np.sqrt(num_nodes)
    return g1

def norm_based_localization_2(obs_vectors, null_max):
    null_max = null_max[:, :obs_vectors.shape[1]] # size: [m, 20]
    m = null_max.shape[0]
    t_a = np.full(obs_vectors.shape, 1.0)  # size: [num_nodes, 20]
    for x in null_max: # x size: [20, ]
        t_a += (obs_vectors <= x)
    t_a /= (m + 1)
    t_a[np.where(t_a>=0.5)] = 0.5
    g2 = norm.ppf(1-t_a)
    g2 = np.clip(g2, np.min(g2), 8)
    return g2

def norm_based_localization_3(obs_vectors, null_mean, null_std):
    null_mean = null_mean[:obs_vectors.shape[1]] # size: [20, ]
    null_std = null_std[:obs_vectors.shape[1]] # size: [20, ]
    # p_value = norm.cdf(obs_vectors, loc=null_mean, scale=null_std)
    # inverse_cdf = norm.ppf(np.real(p_value))
    # inverse_cdf = np.clip(inverse_cdf, min(inverse_cdf), 8)
    inverse_cdf = p_value_test(obs_vectors, null_mean, null_std)
    v_geq_null_mean = np.abs(obs_vectors) >= null_mean
    g3 = np.maximum(inverse_cdf, 0) * v_geq_null_mean
    return g3

def norm_based_localization_4(obs_vectors, null_mean):
    obs_abs = np.abs(obs_vectors)
    null_mean = null_mean[:obs_vectors.shape[1]]
    v_geq_null_mean = np.abs(obs_vectors) >= null_mean
    g4 = obs_abs * v_geq_null_mean
    return g4

def inverse_participation_ratio(vectors):
    """
    Parameters:
    -----------
        vectors: 2D array, each column is a vector
    """
    ipr = np.power(vectors, 4).sum(axis=0)
    return ipr

def exponential_norm(vectors):
    """
    Parameters:
    -----------
        vectors: 2D array, each column is a vector
    """
    v_abs = np.abs(vectors)
    exp_norm = np.exp(v_abs) - v_abs - 1
    return exp_norm.sum(axis=0)

def count_cum_quantile(vector):
    """Counts the minimum number of entries to occupy 90% of the mass of the vector
    """
    v = np.sort(vector)[::-1]
    v_sum = np.sum(v)
    top90 = 0.9 * v_sum
    cum = 0
    count = 0
    while cum < top90:
        cum += v[count]
        count += 1
    while count < len(v) and v[count] == v[count-1]:
        count += 1
    return count

def direct_localisation(vectors):
    ipr_vectors = np.power(vectors, 4)
    ipr90_count = np.apply_along_axis(func1d=count_cum_quantile, axis=0, arr=ipr_vectors)
    
    abs_vectors = np.abs(vectors)
    abs90_count = np.apply_along_axis(func1d=count_cum_quantile, axis=0, arr=abs_vectors)
    
    return ipr90_count, abs90_count

def null_direct_local_params(null_vectors):
    """Gets useful statistic parameters on the null eigenvectors.
    Parameters:
    -----------
        null_vectors: matrix of size: [m, n, 20]
    """
    ipr90_count = np.zeros((len(null_vectors), 20))
    abs90_count = np.zeros((len(null_vectors), 20))
    for m, null_vs in enumerate(null_vectors):
        assert null_vs.shape[1] == 20
        ipr90_count[m, :], abs90_count[m, :] = direct_localisation(null_vs)
    ipr90_count_mean = np.mean(ipr90_count, axis=0)
    ipr90_count_std = np.std(ipr90_count, axis=0)
    abs90_count_mean = np.mean(abs90_count, axis=0)
    abs90_count_std = np.std(abs90_count, axis=0)
    return (ipr90_count_mean, ipr90_count_std), (abs90_count_mean, abs90_count_std)

def direct_score(obs_vectors, null_vectors):
    """
    Returns:
    --------
        ipr90_scores, abs90_scores: shape [num_eigenvectors, ]
    """
    ipr90_count, abs90_count = direct_localisation(obs_vectors)
    (ipr90_mean, ipr90_std), (abs90_mean, abs90_std) = null_direct_local_params(null_vectors)
    ipr90_scores = p_value_test(ipr90_count, ipr90_mean, ipr90_std, clip=0.05)
    abs90_scores = p_value_test(abs90_count, abs90_mean, abs90_std, clip=0.05)
    return ipr90_scores, abs90_scores


def p_value_test(obs, mean_params, std_params, clip=0.5):
    """Compute the inverse standard normal distribution score
    """
    mean_params = mean_params[:len(obs)]
    std_params = std_params[:len(obs)]
    # print(std_params)
    ok_idx_1 = np.where(std_params!=0)[0]
    obs_pvalue = np.zeros(obs.shape, dtype=float)
    obs_pvalue[ok_idx_1] = norm.cdf(obs[ok_idx_1], loc=mean_params[ok_idx_1], scale=std_params[ok_idx_1])
    scores = np.zeros(obs_pvalue.shape, dtype=float)
    ok_idx = np.where(obs_pvalue>1-clip)[0]
    scores[ok_idx] = norm.ppf(obs_pvalue[ok_idx])
    scores = np.clip(scores, a_min=0, a_max=8)
    return scores

def norm_based_stats(obs_g, null_gs, stat='ipr'):
    """computes the p_score
    obs_g: [num_vectors, ]
    null_g: [num_samples, num_vectors]
    """
    if stat == 'ipr':
        obs_stat = inverse_participation_ratio(obs_g)
        null_stats = np.array([inverse_participation_ratio(n_g) for n_g in null_gs])
    elif stat == 'exp':
        obs_stat = exponential_norm(obs_g)
        null_stats = np.array([exponential_norm(n_g) for n_g in null_gs])
    null_mean = np.mean(null_stats, axis=0)
    null_std = np.std(null_stats, axis=0)
    p_score = p_value_test(obs_stat, null_mean, null_std, clip=0.05)
    return p_score

def norm_based_scores(obs_vectors, null_vectors):
    """
    Parameters:
    -----------
        obs_vectors: matrix of eigenvectors, W_sym_upper, W_sym_lower, L_comb, L_rw
            size [num_nodes], 20]
        null_vectors: matrix of eigenvectors of null models, size [m, num_nodes, 20]
    
    Returns:
    --------
        iprs, exps: shape [4, num_eigenvectors]
    """
    null_max, null_mean, null_std = null_norm_based_params(null_vectors)
    # obs_g's are in the same shape of obs_vectors
    obs_g1 = norm_based_localization_1(obs_vectors)
    null_g1 = [norm_based_localization_1(null_v) for null_v in null_vectors]
    obs_g2 = norm_based_localization_2(obs_vectors, null_max=null_max)
    null_g2 = [norm_based_localization_2(null_v, null_max=null_max) for null_v in null_vectors]
    obs_g3 = norm_based_localization_3(obs_vectors, null_mean=null_mean, null_std=null_std)
    null_g3 = [norm_based_localization_3(null_v, null_mean=null_mean, null_std=null_std) for null_v in null_vectors]
    obs_g4 = norm_based_localization_4(obs_vectors, null_mean=null_mean)
    null_g4 = [norm_based_localization_4(null_v, null_mean=null_mean) for null_v in null_vectors]
    obs_gs = [obs_g1, obs_g2, obs_g3, obs_g4]
    null_gs = [null_g1, null_g2, null_g3, null_g4]
    ipr_scores = [norm_based_stats(obs_g, n_gs, stat='ipr') for obs_g, n_gs in zip(obs_gs, null_gs)]
    exp_scores = [norm_based_stats(obs_g, n_gs, stat='exp') for obs_g, n_gs in zip(obs_gs, null_gs)]
    return ipr_scores, exp_scores

def sign_statistic(vectors):
    """
    Returns:
    --------
        sign_stat: shape [num_eigenvectors, ]
        N_pos, N_neg: shape [num_eigenvectors, ]
    """
    theta = min(20, vectors.shape[1])
    N_pos = np.sum(vectors>0, axis=0)
    N_neg = np.sum(vectors<0, axis=0)
    N_pos += (N_pos == 0) * theta
    N_neg += (N_neg == 0) * theta
    sign_stat = np.maximum(N_pos, N_neg) / theta
    return sign_stat, N_pos, N_neg

def null_sign_params(null_vectors):
    """Gets useful statistic parameters on the null eigenvectors.
    Parameters:
    -----------
        null_vectors: matrix of size: [m, n, 20]
    """
    null_sign_stats = np.zeros((len(null_vectors), 20))
    for m, null_vs in enumerate(null_vectors):
        assert null_vs.shape[1] == 20
        null_sign_stats[m, :], _, _ = sign_statistic(null_vs)
    null_mean = np.mean(null_sign_stats, axis=0)
    null_std = np.std(null_sign_stats, axis=0)
    return null_mean, null_std

def sign_based_score(obs_vectors, null_vectors):
    """
    Returns:
    --------
        4 matrix of the same shape of obs_vectors
    """
    obs_sign, N_pos, N_neg = sign_statistic(obs_vectors)
    null_sign_mean, null_sign_std = null_sign_params(null_vectors)
    p_score = p_value_test(obs_sign, null_sign_mean, null_sign_std, clip=0.05)
    sign_stat_1 = ((N_pos < N_neg) * (obs_vectors > 0) + (N_pos > N_neg) * (obs_vectors < 0)) * p_score
    sign_stat_2 = ((N_pos < N_neg) * (obs_vectors > 0) / N_pos + (N_pos > N_neg) * (obs_vectors < 0) / N_neg) * p_score
    sign_stat_equal_1 = (N_pos == N_neg) * ((obs_vectors > 0) + (obs_vectors < 0)) * p_score
    sign_stat_equal_2 = sign_stat_equal_1 / (N_pos + N_neg)
    return sign_stat_1, sign_stat_2, sign_stat_equal_1, sign_stat_equal_2

def compute_spectral_scores_comm(comm, null_matrices):
    num_nodes = len(comm.nodes())
    obs_upper, obs_lower, obs_comb, obs_rw = comm_eigenvectors(comm)
    total_scores = {}

    obs = [obs_upper, obs_lower, obs_comb, obs_rw]
    score_name_prefix = ['upper', 'lower', 'comb', 'rw']
    for obs_vectors, null_vectors, name_prefix in zip(obs, null_matrices, score_name_prefix):
        ipr_scores, exp_scores = norm_based_scores(obs_vectors, null_vectors)
        for i in range(len(ipr_scores)):
            total_scores['{}_ipr_{}'.format(name_prefix, i+1)] = np.repeat(np.sum(ipr_scores[i]), num_nodes)
            total_scores['{}_exp_{}'.format(name_prefix, i+1)] = np.repeat(np.sum(exp_scores[i]), num_nodes)
        ipr90_scores, abs90_scores = direct_score(obs_vectors, null_vectors)
        total_scores['{}_ipr90'.format(name_prefix)] = np.repeat(np.sum(ipr90_scores), num_nodes)
        total_scores['{}_abs90'.format(name_prefix)] = np.repeat(np.sum(abs90_scores), num_nodes)
        sign_1, sign_2, sign_eq_1, sign_eq_2 = sign_based_score(obs_vectors, null_vectors)
        total_scores['{}_sign_stat_1'.format(name_prefix)] = np.sum(sign_1, axis=1)
        total_scores['{}_sign_stat_2'.format(name_prefix)] = np.sum(sign_2, axis=1)
        total_scores['{}_sign_equal_1'.format(name_prefix)] = np.sum(sign_eq_1, axis=1)
        total_scores['{}_sign_equal_2'.format(name_prefix)] = np.sum(sign_eq_2, axis=1)
        total_scores['{}_absolute_value'.format(name_prefix)] = np.sum(np.abs(obs_vectors), axis=1)
    return total_scores

def spectral_features(graph, null_samples, num_samples=500):
    logging.info("partition graph")
    communities = [graph.subgraph(comm_nodes) for comm_nodes in partition_graph(graph) if len(comm_nodes) > 4]
    logging.info("got {} communities".format(len(communities)))
    logging.info("generating null samples")
    assert len(null_samples) >= num_samples
    null_samples = null_samples[:num_samples]
    null_matrices = [[], [], [], []]
    for n_samp in null_samples:
        assert len(n_samp.nodes()) >= 40
        null_u, null_l, null_c, null_r = comm_eigenvectors(n_samp, num_vectors=20)
        null_matrices[0].append(null_u)
        null_matrices[1].append(null_l)
        null_matrices[2].append(null_c)
        null_matrices[3].append(null_r)
    
    for comm_idx, comm in enumerate(communities):
        logging.info("computing spectral scores for community No.{}".format(comm_idx))
        comm_total_scores = compute_spectral_scores_comm(comm, null_matrices)
        nodes = list(comm.nodes())
        for feature_name, feature_scores in comm_total_scores.items():
            for nid, node in enumerate(nodes):
                graph.node[node][feature_name] = feature_scores[nid]
    return graph

# from generator import ER_generator, draw_anomalies
# graph = ER_generator(n=500, p=0.02, seed=None)
# graph = draw_anomalies(graph)
# _, null_samples = generate_null_models(graph, num_models=3, min_size=20)
# graph = spectral_features(graph, null_samples, num_samples=3)
# # print(graph.nodes(data=True))
# all_features = set()
# for node in graph.nodes():
#     node_features = set(dict(graph.node[node]).keys())
#     if len(node_features) != 41:
#         print("node features: ", node_features)
#     all_features |= node_features
# print("all features: ", all_features)
