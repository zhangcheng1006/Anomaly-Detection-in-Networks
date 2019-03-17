from generator import *
from basic_test import basic_test
from com_detection import community_detection
from path_finder import assign_path_score

G = ER_generator(n=400, p=0.01)
G = draw_anomalies(G)
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
G = assign_path_score(G, num_sampler=20, min_path=2, max_path=20, beamsize=1000)
for path_size in range(2, 20):
    nodes = list(range(400))
    for node in nodes:
        feature = G.node[node].get('path_'+str(path_size))
        if feature is not None and feature != 0:
            print("path size is: " + str(path_size))
            print("node is: " + str(node))
            print("score is: " + str(feature))

