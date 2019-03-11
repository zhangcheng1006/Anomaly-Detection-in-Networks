import networkx as nx
import numpy as np
from scipy.stats import norm

def GAW(weights, metric='gaw'):
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
    gaws = []
    for _ in range(n_samples):
        sampled_weights = np.random.choice(all_weights, n_edges, replace=True)
        gaw = GAW(sampled_weights, metric)
        gaws.append(gaw)
    return np.mean(gaws), np.std(gaws)

def compute_p(value, avg, std):
    return 1 - norm.cdf(value, loc=avg, scale=std)

def compute_node_basic(edges, shared_distribution, metric='gaw'):
    node_gaw = GAW(edges, metric)
    n_edges = len(edges)
    mean_gaw, std_gaw = shared_distribution[n_edges]
    p = compute_p(node_gaw, mean_gaw, std_gaw)

    score = 0
    if p < 0.05:
        score = norm.ppf(1 - p)
    return score

def basic_test(G, metric='gaw'):
    """Compute the basic scores of nodes in the graph, and attach the scores to nodes as an attribute `basic_score`.
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

