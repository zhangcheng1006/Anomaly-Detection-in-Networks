"""This module performs basic test on a given graph 
by assigning each node a basic score according to its 
GAW subject to a random distribution (null hypothesis)."""
import networkx as nx
import numpy as np
from scipy.stats import norm

# def GAW(weights, mode="multi"):
#     """Computes the GAW statistics of a set of edge weights.
#     Parameters:
#     ------
#         weights: array-like, a set of weights
    
#     Returns:
#     --------
#         gaw: float, the GAW statistics.
#     """
#     # print(type(weights[0]))
#     ordered_weights = sorted(weights, reverse=True)

#     if mode == 'simple':
#         crop_len = np.ceil(len(weights))
#         gaw = np.prod(ordered_weights[:int(crop_len)])
#         gaw = np.power(gaw, 1./crop_len)
#         return gaw

#     gaws = []    
#     for proportion in [1, 0.1, 0.2]:
#         crop_len = np.ceil(proportion * len(weights))
#         gaw = np.prod(ordered_weights[:int(crop_len)])
#         gaw = np.power(gaw, 1./crop_len)
#         gaws.append(gaw)
#     return gaws

def GAW(weights, mode="multi"):
    """Computes the GAW statistics of a set of edge weights. (Log-Exp version to avoid numerical error)
    Parameters:
    ------
        weights: array-like, a set of weights
    
    Returns:
    --------
        gaw: float, the GAW statistics.
    """
    # print(type(weights[0]))
    ordered_weights = sorted(weights, reverse=True)

    if mode == 'simple':
        crop_len = np.ceil(len(weights))
        gaw = np.sum(np.log(ordered_weights[:int(crop_len)]))
        gaw = np.exp(gaw/crop_len)
        return gaw

    gaws = []    
    for proportion in [1, 0.1, 0.2]:
        crop_len = np.ceil(proportion * len(weights))
        gaw = np.sum(np.log(ordered_weights[:int(crop_len)]))
        gaw = np.exp(gaw/crop_len)
        gaws.append(gaw)
    return gaws

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
        gaws_sample = GAW(sampled_weights)
        for i in range(3):
            gaws[i].append(gaws_sample[i])
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
    node_gaws = GAW(edges)
    for i in range(3):
        p_values[i] = compute_p(node_gaws[i], gaussian_params[i][0], gaussian_params[i][1])
    
    scores = [norm.ppf(1-p) if p<0.05 else 0 for p in p_values]
    return scores

def basic_features(graph, num_samples=10000):
    """Compute the basic scores of all nodes in the graph, and attach the scores to nodes as an attribute `basic_score`.
    Parameters:
    -----------
        G: a networkx.Graph object, the graph to compute basic scores
    
    Returns:
    --------
        scores: dict with nodes as keys and nodes' scores as values
    """
    all_weights = list(nx.get_edge_attributes(graph, 'weight').values())
    degrees = graph.degree()
    degree_values = [value for _, value in degrees]
    degree_avg = np.mean(degree_values)
    degree_std = np.std(degree_values)
    null_params = {degree: monte_carlo_sampler(degree, all_weights, num_samples) for degree in set(degree_values)}

    for node in graph.nodes():
        edge_weights = [data[2] for data in graph.in_edges(node, data='weight')] + [data[2] for data in graph.out_edges(node, data='weight')]
        gaw_scores = compute_node_gaw_scores(edge_weights, null_params)

        graph.node[node]['gaw_score'] = gaw_scores[0]
        graph.node[node]['gaw10_score'] = gaw_scores[1]
        graph.node[node]['gaw20_score'] = gaw_scores[2]
        # standard degree
        standard_degree = (len(edge_weights) - degree_avg) / degree_std
        graph.node[node]['degree_std'] = standard_degree
    return graph

# from generator import ER_generator, draw_anomalies
# graph = ER_generator(n=500, p=0.02, seed=None)
# graph = draw_anomalies(graph)
# graph = basic_features(graph)
# # print(graph.nodes(data=True))
# all_features = set()
# for node in graph.nodes():
#     node_features = set(dict(graph.node[node]).keys())
#     if len(node_features) != 41:
#         print("node features: ", node_features)
#     all_features |= node_features
# print("all features: ", all_features)

