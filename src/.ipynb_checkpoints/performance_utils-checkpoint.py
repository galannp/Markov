import numpy as np
import networkx as nx
from networkx.algorithms.bipartite import generators
import matplotlib.pyplot as plt

def predict(sampling_algo, G, nb_run, nb_iter):
    '''
    Returns an estimate of the population distribution on one graph and after performing
    several runs of a sampling algorithm.

    Parameters
    ----------
    sampling_algo
        The sampling algorithm that is used (e.g. Metropolis, Houdayer algorithm, ...)
    G : nx.Graph
        The Erdos-Renyi random graph
    nb_run : int
        The number of runs of the sampling algorithm
    nb_iter : in
        The number of steps in the sampling algorithm

    Returns
    -------
    x : np.ndarray
        An estimate of the population distrbition
    '''
    return average_estimate(np.array([houdayer(G, )]))