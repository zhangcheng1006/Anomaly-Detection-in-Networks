import numpy as np
import networkx as nx
import pandas as pd

from generator import ER_generator, draw_anomalies
from basic_test import basic_features
from com_detection import community_detection
from spectral_localisation import spectral_features
from NetEMD_shiwen import NetEMD_features
from path_finder import path_features
from utils import generate_null_models

graph = ER_generator(n=500, p=0.01, seed=None)
graph = draw_anomalies(graph)

_, references = generate_null_models(graph, num_models=5, min_size=20)
_, null_samples = generate_null_models(graph, num_models=5, min_size=20)

graph = basic_features(graph)
graph = community_detection(graph, null_samples, num_samples=10)

graph = NetEMD_features(graph, references, null_samples, num_references=5, num_samples=5)
graph = spectral_features(graph, null_samples, num_samples=5)
graph = path_features(graph, num_samples=5)
features = set()
for node in graph.nodes():
    features |= set(graph.node[node].keys())
features = list(features)
# features.remove('type')
print(features)
len(features)

X = pd.DataFrame(columns=features, index=graph.nodes())
for node in graph.nodes():
    for f in features:
        X.loc[node, f] = graph.node[node].get(f, 0)

X.fillna(0, inplace=True)
X.replace([np.inf, -np.inf], 0, inplace=True)