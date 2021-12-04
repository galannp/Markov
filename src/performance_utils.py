import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def generate_population(N : int):
    '''
    Generates a population of N individuals

    Parameters
    ----------
    N : int
        The number of individuals in the population

    Returns
    -------
    x : np.ndarray
        The generated population constituted of two classes
        The two classes are respectively indicated by -1 and +1
    '''
    return 2 * np.random.randint(2, size=N) - 1


def average_estimate(X):
    '''
    Compute the average estimate of the population distribution from a set of estimates

    Parameters
    ----------
    X : np.ndarray
        contains in row all the estimates we computed using a sampling algorithm (MCMC, Houdayer, ...)

    Returns
    -------
    x_hat : np.ndarray
        an average estimate of the population distribution
    '''
    avg_estimage = np.mean(X, axis=0)
    return np.where(avg_estimage > 0, 1, 0)


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
        An estimate of the population distrbution
    '''
    estimates = np.array([sampling_algo(G, nb_iter) for i in range (nb_run)])
    return average_estimate(estimates)


def compute_overlap(x_star : np.ndarray, x : np.ndarray) -> float:
    '''
    Computes the overlap between the ground truth and the current estimate.

    Parameters
    ----------
    x_star : np.ndarray
        Ground truth vector.
    x : np.ndarray
        Current estimate vector.

    Returns
    -------
    q : float
        The overlap between the inputs.
    '''
    return abs(np.inner(x_star, x)) / len(x_star)