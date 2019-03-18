import numpy as np
from utils import generate_null_model, comm_eigenvectors
from generator import ER_generator, draw_anomalies

obs_network = ER_generator()
obs_network = draw_anomalies(obs_network)

