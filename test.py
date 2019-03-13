from generator import *
from basic_test import basic_test

G = ER_generator(n=100, p=0.01)
G = draw_anomalies(G)
print(list(G.nodes()))
scores = basic_test(G)
print(scores)

