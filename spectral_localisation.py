import numpy as np
import scipy
from scipy import sparse
from scipy.stats import norm
import networkx as nx
import logging
from utils import comm_eigenvectors

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def null_norm_based_params(null_vectors):
    """Gets useful statistic parameters on the null eigenvectors.
    Parameters:
    -----------
        null_vectors: matrix of size: [m, n, 20]
    """
    null_max = np.zeros((len(null_vectors), 20))
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
    t_a = np.full(obs_vectors.shape, 1)  # size: [num_nodes, 20]
    for x in null_max: # x size: [20, ]
        t_a += (obs_vectors <= x)
    t_a /= (m + 1)
    g2 = norm.ppf(1-t_a) * (t_a < 0.5)
    return g2

def norm_based_localization_3(obs_vectors, null_mean, null_std):
    null_mean = null_mean[:obs_vectors.shape[1]] # size: [20, ]
    null_std = null_std[:obs_vectors.shape[1]] # size: [20, ]
    p_value = norm.cdf(obs_vectors, loc=null_mean, scale=null_std)
    inverse_cdf = norm.ppf(p_value)
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
    ipr90_scores = p_value_test(ipr90_count, ipr90_mean, ipr90_std)
    abs90_scores = p_value_test(abs90_count, abs90_mean, abs90_std)
    return ipr90_scores, abs90_scores


def p_value_test(obs, mean_params, std_params):
    """Compute the inverse standard normal distribution score
    """
    mean_params = mean_params[:len(obs)]
    std_params = std_params[:len(obs)]
    obs_pvalue = norm.cdf(obs, loc=mean_params, scale=std_params)
    scores = norm.pdf(obs_pvalue)
    return scores

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
    obs_g2 = norm_based_localization_2(obs_vectors, null_max=null_max)
    obs_g3 = norm_based_localization_3(obs_vectors, null_mean=null_mean, null_std=null_std)
    obs_g4 = norm_based_localization_4(obs_vectors, null_mean=null_mean)
    obs_gs = [obs_g1, obs_g2, obs_g3, obs_g4]
    iprs = np.array([inverse_participation_ratio(obs_g) for obs_g in obs_gs])
    exps = np.array([exponential_norm(obs_g) for obs_g in obs_gs])
    assert exps.shape == iprs.shape == (4, obs_vectors.shape[1])
    return iprs, exps

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
    p_score = p_value_test(obs_sign, null_sign_mean, null_sign_std)

    sign_stat_1 = ((N_pos < N_neg) * (obs_vectors > 0) + (N_pos > N_neg) * (obs_vectors < 0)) * p_score
    sign_stat_2 = ((N_pos < N_neg) * (obs_vectors > 0) / N_pos + (N_pos > N_neg) * (obs_vectors < 0) / N_neg) * p_score
    sign_stat_equal_1 = (N_pos == N_neg) * ((obs_vectors > 0) + (obs_vectors < 0)) * p_score
    sign_stat_equal_2 = sign_stat_equal_1 / (N_pos + N_neg)
    return sign_stat_1, sign_stat_2, sign_stat_equal_1, sign_stat_equal_2

def spectral_localization_features(comm, null_comms):
    num_nodes = len(comm.nodes())
    obs_upper, obs_lower, obs_comb, obs_rw = comm_eigenvectors(comm)
    null_upper = []
    null_lower = []
    null_comb = []
    null_rw = []
    for null_comm in null_comms:
        assert len(null_comm.nodes()) >= 40
        null_u, null_l, null_c, null_r = comm_eigenvectors(comm, min_size=20)
        null_upper.append(null_u)
        null_lower.append(null_l)
        null_comb.append(null_c)
        null_rw.append(null_r)
    total_scores = {}

    obs = [obs_upper, obs_lower, obs_comb, obs_rw]
    null = [null_upper, null_lower, null_comb, null_rw]
    score_name_prefix = ['upper', 'lower', 'comb', 'rw']
    for obs_vectors, null_vectors, name_prefix in zip(obs, null, score_name_prefix):
        ipr_scores, exp_scores = norm_based_scores(obs_vectors, null_vectors)
        for i in range(4):
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
