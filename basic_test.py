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

def monte_carlo_sampler(n_edges, all_weights, n_samples=10000, metric='gaw'):
    """A Monte-Carlo simulator to generate random GAW statistics.
    Parameters:
    -----------
        n_edges: int, the degree (in-degree + out-degree) of the node
        all_weights: all edge weights appearing in the graph from which the simulator choose weights from
        n_samples: int, default `10000`. The number of simulations to repreat.
        metric: str, default `gaw`, options `gaw`, `gaw10` and `gaw20`. The type of GAW to compute.
    
    Returns:
    --------
        (avg, std): a tuple consisting of the mean and standard deviation of the simulated GAWs.
    """
    gaws = []
    for _ in range(n_samples):
        sampled_weights = np.random.choice(all_weights, n_edges, replace=True)
        gaw = GAW(sampled_weights, metric)
        gaws.append(gaw)
    return np.mean(gaws), np.std(gaws)

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

def compute_node_basic(edges, shared_distribution, metric='gaw'):
    """Computes the basic score of a node.
    Parameters:
    -----------
        edges: array-like, the edge weights of the node
        shared_distribution: dict, the null Gaussian distributions for different node degrees
        metric: str, default `gaw`, options `gaw`, `gaw10` and `gaw20`. The type of GAW to compute.

    Returns:
    --------
        score: float, the basic score of the node
    """
    node_gaw = GAW(edges, metric)
    n_edges = len(edges)
    mean_gaw, std_gaw = shared_distribution[n_edges]
    p = compute_p(node_gaw, mean_gaw, std_gaw)

    score = 0
    if p < 0.05:
        score = norm.ppf(1 - p)
    return score

def basic_test(G, metric='gaw'):
    """Compute the basic scores of all nodes in the graph, and attach the scores to nodes as an attribute `basic_score`.
    Parameters:
    -----------
        G: a networkx.Graph object, the graph to compute basic scores
        metric: str, default `gaw`, options `gaw`, `gaw10` and `gaw20`. The type of GAW to compute.
    
    Returns:
    --------
        scores: dict with nodes as keys and nodes' scores as values
    """
    all_weights = list(nx.get_edge_attributes(G, 'weight').values())
    degrees = set(G.degrees.values())
    shared_distribution = {degree: monte_carlo_sampler(degree, all_weights, metric=metric) for degree in degrees}

    scores = {}
    for node in G.nodes():
        edge_weights = [data[3] for data in G.in_edges(node, data='weight')] + [data[3] for data in G.out_edges(node, data='weight')]
        score = compute_node_basic(edge_weights, shared_distribution, metric)
        scores[node] = score
        G.node[node]['basic_score'] = score
    return scores

