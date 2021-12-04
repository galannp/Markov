import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy import random
import sys
sys.path.append('../src')

from performance_utils import *


def houdayer_step(G, x1, x2):
    '''
    Performs one step of the Houdayer algorithm

    Parameters
    ----------
    G : nx.Graph
        The Erdos-Renyi random graph
    x1, x2 : np.ndarray
        The current state of the coupled chain

    Returns
    -------
    x1, x2 : np.ndarray
        The state of the coupled chain after performing one step of the algorithm
    '''
    # Local overlapping
    y = x1 * x2
    # Selecting a node at random for which the overlapping is -1
    idx_cluster = random.choice(np.argwhere(y == -1).flatten(), 1)[0]
    # The connex component associated to this node
    cluster_minus_1 = nx.node_connected_component(G, idx_cluster)
    # Flipping the sign of each individual in the connex components
    for node in cluster_minus_1:
        x1[node] *= -1
        x2[node] *= -1
    return x1, x2


def houdayer(G, nb_iter):
    '''
    Performs several steps of the Houdayer algorithm

    Parameters
    ----------
    G : nx.Graph
        The Erdos-Renyi random graph
    nb_iter : int
        The number of steps in the Houdayer algorithm

    Returns
    -------
    x : np.ndarray
        A sample estimate after performing n_iter steps of the algorithm.
    '''
    N = G.number_of_nodes()
    # Initial states for the coupled chain
    x1, x2 = generate_population(N), generate_population(N)
    # n_iter iterations of the algorithm
    for it in range (nb_iter):
        x1, x2 = houdayer_step(G, x1, x2)
    # Returning the average of the two samples x1 and x2
    return average_estimate(np.array([x1, x2]))


def metropolis_houdayer():
    '''
    '''
    # TODO
    pass