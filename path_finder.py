import numpy as np
import heapq as hq
import networkx as nx
from basic_test import compute_p
from scipy.stats import norm

def fitness(path):
    return min(path[1])

def get_base_paths(graph, beamsize=5000):
    h = []
    for node in graph.nodes():
        out_edges = graph.out_edges(node, data='weight')
        if out_edges is None or len(out_edges) == 0:
            continue
        out_edges = sorted(out_edges, key=lambda x: x[2], reverse=True)
        in_edges = graph.in_edges(node, data='weight')
        if in_edges is None or len(in_edges) == 0:
            continue
        in_edges = sorted(in_edges, key=lambda x: x[2], reverse=True)

        currentIn = in_edges.pop(0)
        currentOut = out_edges.pop(0)
        while currentIn is not None and currentOut is not None:
            if currentIn[0] == currentOut[1]:
                if len(out_edges) == 0:
                    if len(in_edges) == 0:
                        break
                    else:
                        currentIn = in_edges.pop(0)
                else:
                    if len(in_edges) == 0:
                        currentOut = out_edges.pop(0)
                    else:
                        if in_edges[0][2] >= out_edges[0][2]:
                            currentIn = in_edges.pop(0)
                        else:
                            currentOut = out_edges.pop(0)
                continue

            path = ([currentIn[0], node, currentOut[1]], [currentIn[2], currentOut[2]])
            fit = fitness(path)
            element = (path, fit)
            # print(element)
            if len(h) < beamsize:
                hq.heappush(h, element)
                enter = True
            else:
                smallest = hq.heappushpop(h, element)
                if smallest[1] < element[1]:
                    enter = True
                else:
                    enter = False
            if enter:
                if len(out_edges) == 0:
                    if len(in_edges) == 0:
                        break
                    else:
                        currentIn = in_edges.pop(0)
                else:
                    if len(in_edges) == 0:
                        currentOut = out_edges.pop(0)
                    else:
                        if in_edges[0][2] >= out_edges[0][2]:
                            currentIn = in_edges.pop(0)
                        else:
                            currentOut = out_edges.pop(0)
            else:
                break

    return h


def get_next_size_paths(graph, paths, beamsize=5000):
    h = []
    for path in paths:
        last_node = path[0][0][-1]
        for neighbor in graph.neighbors(last_node):
            if neighbor not in path[0][0]:
                weight = graph.adj[last_node][neighbor]['weight']
                extended_path = (path[0][0] + [neighbor], path[0][1] + [weight])
                if weight < path[1]:
                    element = (extended_path, weight)
                else:
                    element = (extended_path, path[1])
                if len(h) < beamsize:
                    hq.heappush(h, element)
                else:
                    hq.heappushpop(h, element)
    return h



def get_null_distribution(graph, num_sampler=500, min_path=2, max_path=20, beamsize=5000):
    g = graph.copy()
    all_weights = list(nx.get_edge_attributes(g, 'weight').values())
    all_largest_weights = np.zeros((max_path - min_path + 1, num_sampler))
    for num in range(num_sampler):
        np.random.shuffle(all_weights)
        for i, edge in enumerate(g.edges()):
            g.adj[edge[0]][edge[1]]['weight'] = all_weights[i]
            
        # print("NULL DISTRIBUTION: Finding the 2th paths...")
        paths = get_base_paths(g, beamsize)
        largest_weight = 0
        for path in paths:
            weight = sum(path[0][1])
            if weight > largest_weight:
                largest_weight = weight
        all_largest_weights[0, num] = largest_weight
        for path_size in range(max_path - min_path):
            # print("NULL DISTRIBUTION: Finding the " + str(path_size+3) + "th paths...")
            paths = get_next_size_paths(graph, paths)
            largest_weight = 0
            for path in paths:
                weight = sum(path[0][1])
                if weight > largest_weight:
                    largest_weight = weight
            all_largest_weights[path_size + 1, num] = largest_weight
    null_mean = np.mean(all_largest_weights, axis=1)
    null_std = np.std(all_largest_weights, axis=1)
    return null_mean, null_std


def assign_path_score(graph, num_sampler=500, min_path=2, max_path=20, beamsize=5000):
    null_mean, null_std = get_null_distribution(graph, num_sampler, min_path, max_path, beamsize)
    for path_size in range(min_path, max_path+1):
        print("Find all " + str(path_size) + "th paths...")
        if path_size == 2:
            paths = get_base_paths(graph, beamsize)
        else:
            paths = get_next_size_paths(graph, paths, beamsize)
        
        for path in paths:
            weight = sum(path[0][1])
            p_value = compute_p(weight, null_mean[path_size-2], null_std[path_size-2])
            if p_value >= 0.5:
                score = 0
            else:
                score = norm.ppf(1 - p_value)
            for node in path[0][0]:
                if graph.node[node].get('path_'+str(path_size)) is None:
                    graph.node[node]['path_'+str(path_size)] = score
                elif graph.node[node]['path_'+str(path_size)] < score:
                    graph.node[node]['path_'+str(path_size)] = score
    
    return graph

