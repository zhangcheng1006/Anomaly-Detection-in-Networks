from generator import *
from basic_test import basic_test
from com_detection import community_detection

G = ER_generator(n=400, p=0.01)
G = draw_anomalies(G)
# print(list(G.nodes()))
# scores = basic_test(G)
# # print(scores)

# print(scores[0])
# print(scores[399])
# print(G.node[1]['gaw_score'])
# print(G.node[10]['gaw10_score'])
# print(G.node[100]['gaw20_score'])
# print(G.node[399]['degree_std'])

G = community_detection(G)
print(G.node[1]['first_density'])
print(G.node[10]['second_density'])
print(G.node[100]['third_density'])
print(G.node[50]['small_community'])
print(G.node[200]['first_strength'])
print(G.node[399]['second_strength'])

