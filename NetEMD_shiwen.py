import numpy as np
import networkx as nx

def direction_eigenvectors(vectors):
    """Normalize the direction of the eigenvectors according to Appendix.D
    vectors is a 2D matrix, each column an eigen vector
    """
    num_pos = np.sum(vectors>0, axis=0)
    num_neg = np.sum(vectors<0, axis=0)
    direction = num_pos >= num_neg
    vectors *= direction
    equal_cols = np.where(num_pos == num_neg)
    for eq_col in equal_cols:
        v = vectors[:, eq_col].copy()
        v_2 = np.power(v, 2)
        while True:
            if sum(v) < 0:
                vectors[:, eq_col] *= -1
                break
            elif sum(v) > 0:
                break
            v *= v_2
    return vectors


