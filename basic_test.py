"""This module performs basic test on a given graph 
by assigning each node a basic score according to its 
GAW subject to a random distribution (null hypothesis)."""
import networkx as nx
import numpy as np
from scipy.stats import norm

def GAW(weights, metric='gaw'):
    """Computes the GAW statistics of a set of edge weights.
    Parameters:
    ------
        weights: array-like, a set of weights
        metric: str, default `gaw`, options `gaw`, `gaw10` and `gaw20`. The type of GAW to compute.
    
    Returns:
    --------
        gaw: float, the GAW statistics.
    """
    ordered_weights = sorted(weights, reverse=True)
    proportion = len(weights)
    if metric in ['gaw10', 'GAW10']:
        proportion = np.ceil(0.1 * len(weights))
    elif metric in ['gaw20', 'GAW20']:
        proportion = np.ceil(0.2 * len(weights))
    prod = np.prod(ordered_weights[:proportion])
    gaw = np.power(prod, 1./proportion)
    return gaw

def monte_carlo_sampler(n_edges, all_weights, n_samples=10000):
    """A Monte-Carlo simulator to generate random GAW statistics.
    Parameters:
    -----------
        n_edges: int, the degree (in-degree + out-degree) of the node
        all_weights: all edge weights appearing in the graph from which the simulator choose weights from
        n_samples: int, default `10000`. The number of simulations to repreat.
    
    Returns:
    --------
        null_params: a list of tuples consisting of the mean and standard deviation of the simulated GAWs.
    """
    gaws = [[], [], []]
    for _ in range(n_samples):
        sampled_weights = np.random.choice(all_weights, n_edges, replace=True)
        gaw = GAW(sampled_weights, metric='gaw')
        gaws[0].append(gaw)
        gaw10 = GAW(sampled_weights, metric='gaw10')
        gaws[1].append(gaw10)
        gaw20 = GAW(sampled_weights, metric='gaw20')
        gaws[2].append(gaw20)
    return [(np.mean(gaws[i]), np.std(gaws[i])) for i in range(3)]

def compute_p(value, avg, std):
    """Computes the p-value of a specific value to a Gaussian distribution.
    Parameters:
    -----------
        value: float, the value whose p-value to be computed
        avg: the mean of the Gaussian distribution
        std: the standard deviation of the Gaussian distribution
    
    Returns:
    --------
        p_value: float, the computed p-value
    """
    return 1 - norm.cdf(value, loc=avg, scale=std)

def compute_node_gaw_scores(edges, null_params):
    """Computes the basic score of a node.
    Parameters:
    -----------
        edges: array-like, the edge weights of the node
        shared_distribution: dict, the null Gaussian distributions for different node degrees

    Returns:
    --------
        scores: list of float, the basic scores of the node
    """
    n_edges = len(edges)
    gaussian_params = null_params[n_edges]

    p_values = [0, 0, 0]
    for i, metric in enumerate(['gaw', 'gaw10', 'gaw20']):
        node_gaw = GAW(edges, metric)
        p_values[i] = compute_p(node_gaw, gaussian_params[i][0], gaussian_params[i][1])
    
    scores = [norm.ppf(1-p) if p<0.05 else 0 for p in p_values]
    return scores

def basic_test(G):
    """Compute the basic scores of all nodes in the graph, and attach the scores to nodes as an attribute `basic_score`.
    Parameters:
    -----------
        G: a networkx.Graph object, the graph to compute basic scores
    
    Returns:
    --------
        scores: dict with nodes as keys and nodes' scores as values
    """
    all_weights = list(nx.get_edge_attributes(G, 'weight').values())
    degrees = G.degree()
    degree_values = degrees.values()
    degree_avg = np.mean(degree_values)
    degree_std = np.std(degree_values)
    null_params = {degree: monte_carlo_sampler(degree, all_weights) for degree in set(degree_values)}

    scores = {}
    for node in G.nodes():
        node_score_dict = {}
        edge_weights = [data[3] for data in G.in_edges(node, data='weight')] + [data[3] for data in G.out_edges(node, data='weight')]
        gaw_scores = compute_node_gaw_scores(edge_weights, null_params)
        node_score_dict['gaw_score'] = gaw_scores[0]
        G.node[node]['gaw_score'] = gaw_scores[0]
        node_score_dict['gaw10_score'] = gaw_scores[1]
        G.node[node]['gaw10_score'] = gaw_scores[1]
        node_score_dict['gaw20_score'] = gaw_scores[2]
        G.node[node]['gaw20_score'] = gaw_scores[2]
        # standard degree
        standard_degree = (len(edge_weights) - degree_avg) / degree_std
        node_score_dict['degree_std'] = standard_degree
        G.node[node]['degree_std'] = standard_degree
        scores[node] = node_score_dict
    return scores

