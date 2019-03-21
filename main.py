import numpy as np
import pandas as pd
import networkx as nx
import logging
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

from utils import generate_null_models
from generator import ER_generator, draw_anomalies
from basic_test import basic_features
from com_detection import community_detection
from spectral_localisation import spectral_features
from NetEMD_shiwen import NetEMD_features
from path_finder import path_features

parameters = [(0.01, 1.0), (0.01, 0.998), (0.01, 0.996), 
              (0.01, 0.994), (0.02, 1.0), (0.02, 0.998), 
              (0.03, 1.0), (0.03, 0.998)]

num_models_per_param = 10
num_nodes = 2000
num_basic_mc_samples = 1000
num_references = 15
num_null_models = 100

for param_id, (p, w) in enumerate(parameters[:4]):
    references = None
    null_samples_whole = None
    null_samples = None
    for model_id in range(num_models_per_param):
        logging.info("Computing {}-th parameter combination (p={}, w={})".format(param_id, p, w))
        logging.info("Computing {}-th/{} model".format(model_id, num_models_per_param))
        graph = ER_generator(n=num_nodes, p=p, seed=None)
        graph = draw_anomalies(graph, w=w)
        if model_id == 0:
            _, references = generate_null_models(graph, num_models=num_references, min_size=20)
            null_samples_whole, null_samples = generate_null_models(graph, num_models=num_null_models, min_size=20)

        logging.info("\n\nGenerating basic features\n\n")
        graph = basic_features(graph, num_samples=num_basic_mc_samples)
        logging.info("\n\nGenerating community features\n\n")
        graph = community_detection(graph, null_samples, num_samples=20)
        logging.info("\n\nGenerating NetEMD features\n\n")
        graph = NetEMD_features(graph, references, null_samples, num_references=num_references, num_samples=num_null_models)
        logging.info("\n\nGenerating spectral features\n\n")
        graph = spectral_features(graph, null_samples, num_samples=num_null_models)
        logging.info("\n\nGenerating path features\n\n")
        graph = path_features(graph, null_samples_whole, num_samples=num_null_models)
        features = set()
        for node in graph.nodes():
            features |= set(graph.node[node].keys())
        # features.remove('type')
        logging.info("\n\nComposing DataFrame\n\n")
        X = pd.DataFrame(columns=features, index=graph.nodes())
        for node in graph.nodes():
            for f in features:
                X.loc[node, f] = graph.node[node].get(f, 0)
        X.fillna(0, inplace=True)
        X.replace([np.inf, -np.inf], 0, inplace=True)
        logging.info("\n\nWriting to local file\n\n")
        X.to_csv('Network_p_{}_w_{}_{}.csv'.format(p, w, model_id))

