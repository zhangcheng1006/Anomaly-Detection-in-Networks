import networkx as nx
import numpy as np
import community
from basic_test import compute_p, GAW
from scipy.stats import norm

def percentile(graph, q=99):
    all_weights = list(nx.get_edge_attributes(graph, 'weight').values())
    return np.percentile(all_weights, q)

def augmentation(graph, threshold):
    g = graph.copy()
    n = g.number_of_nodes()
    while True:
        finish = True
        for i in range(n):
            neighbors = g.neighbors(i)
            for neighbor in neighbors:
                w1 = g.get_edge_data(i, neighbor)['weight']
                if w1 > threshold:
                    hop2neighbors = g.neighbors(neighbor)
                    for hop2 in hop2neighbors:
                        w2 = g.get_edge_data(neighbor, hop2)['weight']
                        if hop2 != i and w2 > threshold:
                            if g.get_edge_data(hop2, i) is None:
                                g.add_edge(hop2, i, weight=min([w1, w2]))
                                finish = False
                            else:
                                w3 = g.get_edge_data(hop2, i)['weight']
                                if w3 < w1 and w3 < w2:
                                    g[hop2][i]['weight'] = min(w1, w2)
                                    finish = False
        if finish:
            break
    return g

def get_partition(graph):
    #first compute the best partition
    partition = community.best_partition(graph.to_undirected())
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

def get_null_distribution(graph, num_sampler):
    g = graph.copy()
    all_weights = list(nx.get_edge_attributes(g, 'weight').values())
    all_densities = []
    for num in range(num_sampler):
        np.random.shuffle(all_weights)
        for i, edge in enumerate(g.edges()):
            g.adj[edge[0]][edge[1]]['weight'] = all_weights[i]
        communities = get_partition(g.to_undirected())
        choice = np.random.randint(len(communities.keys()))
        all_densities.append(nx.density(communities[choice]))
    mean = np.mean(all_densities)
    std = np.std(all_densities)
    return mean, std

def compute_third_density(graph, communities, num_sampler):
    mean, std = get_null_distribution(graph, num_sampler)
    for com, sub_g in communities.items():
        density = nx.density(sub_g)
        p_value = compute_p(density, mean, std)
        if p_value >= 0.05:
            score = 0
        else:
            score = norm.ppf(1 - p_value)
        for node in sub_g.nodes():
            graph.node[node]['third_density'] = score
    return graph

def small_community_feature(graph, communities, criterion):
    for node in graph.nodes():
        graph.node[node]['small_community'] = 0
    for com, sub_g in communities.items():
        num_nodes = sub_g.number_of_nodes()
        if num_nodes <= criterion:
            for node in sub_g.nodes():
                graph.node[node]['small_community'] = 1
    return graph

def compute_first_strength(graph, communities):
    all_weights = list(nx.get_edge_attributes(graph, 'weight').values())
    network_gaw = GAW(all_weights, mode='simple')
    for com, sub_g in communities.items():
        weights = list(nx.get_edge_attributes(sub_g, 'weight').values())
        com_gaw = GAW(weights, mode='simple')
        strength = com_gaw / network_gaw
        for node in sub_g.nodes():
            graph.node[node]['first_strength'] = strength
    return graph

def compute_second_strength(graph, communities):
    for com, sub_g in communities.items():
        num_nodes = sub_g.number_of_nodes()
        for node in sub_g.nodes():
            graph.node[node]['second_strength'] = graph.node[node]['first_strength'] / num_nodes
    return graph

def community_detection(graph, num_sampler=20, small_criterion=4):
    threshold = percentile(graph)
    augmented_g = augmentation(graph, threshold)
    communities = get_partition(augmented_g)
    graph = compute_first_density(graph, communities)
    graph = compute_second_density(graph, communities)
    graph = compute_third_density(graph, communities, num_sampler)
    graph = small_community_feature(graph, communities, small_criterion)
    graph = compute_first_strength(graph, communities)
    graph = compute_second_strength(graph, communities)
    return graph

