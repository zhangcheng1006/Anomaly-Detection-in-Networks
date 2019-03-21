from generator import ER_generator, draw_anomalies
from basic_test import basic_features
from com_detection import community_detection
from path_finder import path_features

# G = ER_generator(n=400, p=0.01)
# G = draw_anomalies(G)
# print(list(G.nodes()))

# ################ test basic node score #################
# scores = basic_test(G)
# # print(scores)
# print(scores[0])
# print(scores[399])
# print(G.node[1]['gaw_score'])
# print(G.node[10]['gaw10_score'])
# print(G.node[100]['gaw20_score'])
# print(G.node[399]['degree_std'])

# ############### test community detection ###############
# G = community_detection(G)
# print(G.node[1]['first_density'])
# print(G.node[10]['second_density'])
# print(G.node[100]['third_density'])
# print(G.node[50]['small_community'])
# print(G.node[200]['first_strength'])
# print(G.node[399]['second_strength'])

################# test path finder ###################
# G = assign_path_score(G, num_sampler=20, min_path=2, max_path=20, beamsize=1000)
# for path_size in range(2, 20):
#     nodes = list(range(400))
#     for node in nodes:
#         feature = G.node[node].get('path_'+str(path_size))
#         if feature is not None and feature != 0:
#             print("path size is: " + str(path_size))
#             print("node is: " + str(node))
#             print("score is: " + str(feature))

############## NetEMD ##################################
from utils import *
from NetEMD_shiwen import compute_matrix_stat

graph = ER_generator(n=500, p=0.01, seed=None)

# comm_nodes = partition_graph(graph)
# for idx, comm in enumerate(comm_nodes):
    # print("community No.{}/{}".format(idx+1, len(comm_nodes)))
    # comm_graph = graph.subgraph(comm)
    # w_u, w_l, comb, rw = comm_eigenvectors(comm_graph, num_vectors=5)
    # matrix_stats = compute_matrix_stat(comm_graph, normalize=True)
    # print("W-upper", w_u)
    # print("W lower", w_l)
    # print("comb", comb)
    # print("rw", rw)

# _, null_samples = generate_null_models(graph, num_models=1, min_size=5)
# print(type(null_samples), type(null_samples[0]))
# for sample in null_samples:
#     matrix_stats = compute_matrix_stat(sample, normalize=True)
#     w_u, w_l, comb, rw = comm_eigenvectors(sample, num_vectors=5)
#     w_u_stat = compute_matrix_stat(w_u, normalize=True)
#     print("W-upper", w_u_stat)
#     w_l_stat = compute_matrix_stat(w_l, normalize=True)
#     print("W lower", w_l_stat)
#     comb_stat = compute_matrix_stat(comb, normalize=True)
#     print("comb", comb_stat)
#     rw_stat = compute_matrix_stat(rw, normalize=True)
#     print("rw", rw_stat)

