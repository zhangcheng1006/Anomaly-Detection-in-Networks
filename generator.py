import networkx as nx
import numpy as np

def ER_generator(n=10000, p=0.001, seed=2019):
    ER = nx.DiGraph()
    edges = []
    np.random.seed(seed)

    for i in range(n):
        nodes = list(range(i)) + list(range(i+1, n))
        np.random.shuffle(nodes)
        for e in nodes:
            eps = np.random.rand()
            if eps < p:
                w = np.random.rand()
                edges.append((i, e, w))

    ER.add_weighted_edges_from(edges)
    return ER

def add_rings(graph, sizes, w):
    n = graph.number_of_nodes()
    nodes = np.array(range(n))
    np.random.shuffle(nodes)
    begin_index = 0
    for i, size in enumerate(sizes):
        nodes_to_add = np.append(nodes[begin_index:begin_index+size], nodes[begin_index])
        for j in range(len(nodes_to_add) - 1):
            weight = np.random.rand() * (1 - w) + w
            graph.add_weighted_edges_from([(nodes_to_add[j], nodes_to_add[j+1], weight)])
        begin_index += size
    return graph


def add_paths(graph, sizes, w):
    n = graph.number_of_nodes()
    nodes = np.array(range(n))
    np.random.shuffle(nodes)
    begin_index = 0
    for i, size in enumerate(sizes):
        nodes_to_add = nodes[begin_index:begin_index+size]
        for j in range(len(nodes_to_add) - 1):
            weight = np.random.rand() * (1 - w) + w
            graph.add_weighted_edges_from([(nodes_to_add[j], nodes_to_add[j+1], weight)])
        begin_index += size
    return graph

def add_stars(graph, sizes, w):
    n = graph.number_of_nodes()
    nodes = np.array(range(n))
    np.random.shuffle(nodes)
    begin_index = 0
    for i, size in enumerate(sizes):
        center = nodes[begin_index]
        stars = nodes[begin_index+1:begin_index+size]
        for j in range(len(stars)):
            weight = np.random.rand() * (1 - w) + w
            eps = np.random.rand()
            if eps < 0.5:
                graph.add_weighted_edges_from([(center, stars[j], weight)])
            else:
                graph.add_weighted_edges_from([(stars[j], center, weight)])
        begin_index += size
    return graph

def add_cliques(graph, sizes, w):
    n = graph.number_of_nodes()
    nodes = np.array(range(n))
    np.random.shuffle(nodes)
    begin_index = 0
    for i, size in enumerate(sizes):
        nodes_to_add = nodes[begin_index:begin_index+size]
        for j in range(len(nodes_to_add)):
            for k in range(len(nodes_to_add)):
                if j != k:
                    weight = np.random.rand() * (1 - w) + w
                    eps = np.random.rand()
                    if eps < 0.5:
                        graph.add_weighted_edges_from([(nodes_to_add[j], nodes_to_add[k], weight)])
                    else:
                        graph.add_weighted_edges_from([(nodes_to_add[k], nodes_to_add[j], weight)])
        begin_index += size
    return graph

def add_trees(graph, num, w, left=5, middle=3, right=1, omega=1):
    n = graph.number_of_nodes()
    nodes = np.array(range(n))
    np.random.shuffle(nodes)
    size = left + middle + right
    for i in range(num):
        nodes_to_add = nodes[i*size:(i+1)*size]
        left_nodes = nodes_to_add[0:left]
        middle_nodes = nodes_to_add[left:left+middle]
        right_nodes = nodes_to_add[left+middle:left+middle+right]
        for l in left_nodes:
            for m in middle_nodes:
                weight = np.random.rand() * (1 - w) + w
                eps = np.random.rand()
                if eps < omega:
                    graph.add_weighted_edges_from([(l, m, weight)])
                else:
                    graph.add_weighted_edges_from([(m, l, weight)])
        for m in middle_nodes:
            for r in right_nodes:
                weight = np.random.rand() * (1 - w) + w
                eps = np.random.rand()
                if eps < omega:
                    graph.add_weighted_edges_from([(m, r, weight)])
                else:
                    graph.add_weighted_edges_from([(r, m, weight)])
        return graph

def draw_anomalies(graph, w=0.99, n_min=5, n_max=21):
    n = np.random.randint(low=n_min, high=n_max)
    anomaly_type = np.random.randint(5)
    sizes = []
    for i in range(n):
        sizes.append(np.random.randint(low=n_min, high=n_max))
    # 0: rings
    if anomaly_type == 0:
        anomaly_graph = add_rings(graph, sizes, w)
    # 1: paths
    elif anomaly_type == 1:
        anomaly_graph = add_paths(graph, sizes, w)
    # 2: cliques
    elif anomaly_type == 2:
        anomaly_graph = add_cliques(graph, sizes, w)
    # 3: stars
    elif anomaly_type == 3:
        anomaly_graph = add_stars(graph, sizes, w)
    # 4: trees
    else:
        anomaly_graph = add_trees(graph, n, w)
    return anomaly_graph

# test
er = ER_generator()
sizes = [5, 10, 8, 7, 20, 17]
er_rings = add_rings(er, sizes, w=0.99)
er_paths = add_paths(er, sizes, w=0.99)
er_stars = add_stars(er, sizes, w=0.99)
er_cliques = add_cliques(er, sizes, w=0.999)
er_trees = add_trees(er, 6, w=0.999)

er_random = draw_anomalies(er)
