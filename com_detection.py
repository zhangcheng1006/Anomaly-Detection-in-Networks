import networkx as nx
import numpy as np
import community
from basic_test import compute_p, GAW
from scipy.stats import norm
from utils import to_undirected_graph, augmentation, percentile

def get_partition(graph):
    #first compute the best partition
    partition = community.best_partition(to_undirected_graph(graph))
    # print(partition)
    num_communities = float(len(set(partition.values())))
    communities = {}
    for key, value in partition.items():
        if communities.get(value) is None:
            sub_g = nx.DiGraph()
            sub_g.add_node(key)
            communities[value] = sub_g
        else:
            sub_g = communities[value]
            sub_g.add_node(key)
            nodes = sub_g.nodes()
            for node in nodes:
                if node != key:
                    edge = graph.get_edge_data(key, node)
                    if edge is not None:
                        sub_g.add_edge(key, node, weight=edge['weight'])
                    edge = graph.get_edge_data(node, key)
                    if edge is not None:
                        sub_g.add_edge(node, key, weight=edge['weight'])
            communities[value] = sub_g
    return communities

def compute_first_density(graph, communities):
    full_density = nx.density(graph)
    for com, sub_g in communities.items():
        density = nx.density(sub_g)
        stat = density / full_density
        for node in sub_g.nodes():
            graph.node[node]['first_density'] = stat
    return graph

def compute_second_density(graph, communities):
    for com, sub_g in communities.items():
        num_nodes = sub_g.number_of_nodes()
        for node in sub_g.nodes():
            graph.node[node]['second_density'] = graph.node[node]['first_density'] / num_nodes
    return graph
    # full_density = nx.density(graph)
    # for com, sub_g in communities.items():
    #     num_nodes = sub_g.number_of_nodes()
    #     density = nx.density(sub_g)
    #     stat = density / full_density / num_nodes
    #     for node in sub_g.nodes():
    #         graph.node[node]['second_density'] = stat
    # return graph

# TODO

def get_null_distribution(graph, null_samples):
    all_densities = []
    for comm in null_samples:
        try:
            all_densities.append(nx.density(comm))
        except:
            print(type(comm))
            exit()
    mean = np.mean(all_densities)
    std = np.std(all_densities)
    return mean, std

def compute_third_density(graph, communities, null_samples):
    mean, std = get_null_distribution(graph, null_samples)
    for _, sub_g in communities.items():
        density = nx.density(sub_g)
        p_value = compute_p(density, mean, std)
        if p_value >= 0.5:
            score = 0
        else:
            score = norm.ppf(1 - p_value)
        score = np.clip(score, a_min=-8, a_max=8)
        for node in sub_g.nodes():
            graph.node[node]['third_density'] = score
    return graph

def small_community_feature(graph, communities, criterion):
    for node in graph.nodes():
        graph.node[node]['small_community'] = 0
    for _, sub_g in communities.items():
        num_nodes = sub_g.number_of_nodes()
        if num_nodes <= criterion:
            for node in sub_g.nodes():
                graph.node[node]['small_community'] = 1
    return graph

def compute_first_strength(graph, communities):
    all_weights = list(nx.get_edge_attributes(graph, 'weight').values())
    network_gaw = GAW(all_weights, mode='simple')
    for _, sub_g in communities.items():
        weights = list(nx.get_edge_attributes(sub_g, 'weight').values())
        com_gaw = GAW(weights, mode='simple')
        strength = com_gaw / network_gaw
        for node in sub_g.nodes():
            graph.node[node]['first_strength'] = strength
    return graph

def compute_second_strength(graph, communities):
    for _, sub_g in communities.items():
        num_nodes = sub_g.number_of_nodes()
        for node in sub_g.nodes():
            graph.node[node]['second_strength'] = graph.node[node]['first_strength'] / num_nodes
    return graph

def community_detection(graph, null_samples, num_samples=20, small_criterion=4):
    augmented_g = augmentation(graph)
    communities = get_partition(augmented_g)
    graph = compute_first_density(graph, communities)
    graph = compute_second_density(graph, communities)
    graph = compute_third_density(graph, communities, null_samples[:num_samples])
    graph = small_community_feature(graph, communities, small_criterion)
    graph = compute_first_strength(graph, communities)
    graph = compute_second_strength(graph, communities)
    return graph

# from generator import ER_generator, draw_anomalies
# from utils import generate_null_models
# graph = ER_generator(n=500, p=0.02, seed=None)
# graph = draw_anomalies(graph)
# _, null_samples = generate_null_models(graph, num_models=4, min_size=10)
# graph = community_detection(graph, null_samples, num_samples=4)
# print(graph.nodes(data=True))
# print('FINISH!')

