import rpy2
print(rpy2.__version__)

import rpy2.situation
for row in rpy2.situation.iter_info():
    print(row)

# import rpy2's package module
import rpy2.robjects.packages as rpackages

# import R's utility package
utils = rpackages.importr('utils')

# select a mirror for R packages
utils.chooseCRANmirror(ind=1) # select the first mirror in the list

# # R package names
# packnames = ('ggplot2', 'devtools', 'netdist')

# # R vector of strings
# from rpy2.robjects.vectors import StrVector

# # Selectively install what needs to be install.
# # We are fancy, just because we can.
# names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
# print(names_to_install)
# if len(names_to_install) > 0:
#     utils.install_packages(StrVector(names_to_install))

import rpy2.robjects as robjects
# install = robjects.r['''devtools::install_github("alan-turing-institute/network-comparison")''']

netdist = rpackages.importr('netdist')

import numpy as np
import scipy.stats as stats

motif_dict = {  ((0, 2), (1, 0), (1, 0)): 4, 
                ((0, 1), (0, 1), (2, 0)): 5, 
                ((0, 1), (1, 0), (1, 1)): 6, 
                ((0, 1), (1, 1), (2, 1)): 7, 
                ((1, 0), (1, 1), (1, 2)): 8, 
                ((1, 1), (1, 1), (2, 2)): 9, 
                ((0, 2), (1, 1), (2, 0)): 10, 
                ((1, 1), (1, 1), (1, 1)): 11, 
                ((0, 2), (2, 1), (2, 1)): 12, 
                ((1, 2), (1, 2), (2, 0)): 13, 
                ((1, 1), (1, 2), (2, 1)): 14, 
                ((1, 2), (2, 1), (2, 2)): 15, 
                ((2, 2), (2, 2), (2, 2)): 16}

def strength_stats(graph):
    for node in graph.nodes():
        graph.node[node]['in_strength'] = sum([data[2] for data in graph.in_edges(node, data='weight')])
        graph.node[node]['out_strength'] = sum([data[2] for data in graph.out_edges(node, data='weight')])
        graph.node[node]['in_out_strength'] = graph.node[node]['in_strength'] + graph.node[node]['out_strength']
    return graph

def get_motif(g, motif_dict):
    in_degrees = [in_degree for node, in_degree in g.in_degree()]
    out_degrees = [out_degree for node, out_degree in g.out_degree()]
    res = sorted(zip(in_degrees, out_degrees), key=lambda x: (x[0], x[1]))
    motif = motif_dict.get(tuple(res))
    return motif

def motif_stats(graph, num_motif=13):
    all_motif_stats = np.zeros(num_motif)
    for node in graph.nodes():
        hop1 = set(graph.in_edges(node, data='weight'))|set(graph.out_edges(node, data='weight'))
        for _, a in hop1:
            hop2 = set(graph.in_edges(a, data='weight'))|set(graph.out_edges(a, data='weight'))|hop1
            for _, b in hop2:
                if a < b:
                    sub_g = graph.subgraph([node, a, b])
                    motif = get_motif(sub_g)
                    stat = np.prod([data[2] for data in sub_g.edges.data('weight', default=1)])
                    all_motif_stats[motif-4] += stat
        for i in range(num_motif):
            graph.node[node]['motif_'+str(i+1)+'_stat'] = all_motif_stats[i]
    return graph

def compute_hist(g, stat):
    T_stats = [data[1] for data in g.nodes.data(stat, default=0)]
    return np.histogram(T_stats, bins='auto', density=True)

def compute_NetEMD(g1, g2, stat):
    h1 = compute_hist(g1, stat)
    h2 = compute_hist(g2, stat)
    dhist1 = netdist.dhist(h1[1], h1[0])
    dhist2 = netdist.dhist(h2[1], h2[0])
    dist = netdist.net_emd(dhist1, dhist2)

def assign_NetEMD_score(graph, num_references=15, num_samples=500):




# x = np.linspace(0, 10, 1000)
# y = np.linspace(1, 11, 1000)

# u1 = robjects.FloatVector(stats.norm.pdf(x))
# u2 = robjects.FloatVector(stats.expon.pdf(y))

# print(netdist.net_emd(u1, u2))