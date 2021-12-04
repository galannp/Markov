import numpy as np
import networkx as nx
from networkx.algorithms.bipartite import generators
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
    Compute the average estimate of the population distribution from a list of estimates

    Parameters
    ----------
    X : list of np.ndarray
        List that contains all the estimate we computed using a sampling algorithm (MCMC, Houdayer, ...)

    Returns
    -------
    x_hat : np.ndarray
        an average estimate of the population distribution
    '''
    avg_estimage = np.mean(X, axis=0)
    return np.where(avg_estimage > 0, 1, 0)


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


def build_graph(x_star : np.ndarray, a : float, b : float) -> nx.Graph:
    '''
    Constructs a networkx graph from from the ground truth and connectivity parameters.

    Parameters
    ----------
    x_star : np.ndarray
        Ground truth vector.
    a : float
        Connectivity parameter for nodes in the same community.
    b : float
        Connectivity parameter for nodes in different communities.

    Returns
    -------
    G : nx.Graph
        The inhomogeneous binomial graph with two classes.
    '''
    # Population
    N = len(x_star)
    N_red = sum(x_star > 0)
    N_blue = N - N_red
    # Probability of edge creation
    p_same = a / N # Probability of edge creating within each class
    p_diff = b / N # Probability of edge creation between two classes

    # Building the graph for each subpopulation
    red_graph = nx.fast_gnp_random_graph(N_red, p_same)
    blue_graph = nx.fast_gnp_random_graph(N_blue, p_same)
    # Building the whole graph by linking the two subpopulations
    main_graph = generators.random_graph(N_red, N_blue, p_diff)
    main_graph.add_edges_from(red_graph.edges())
    main_graph.add_edges_from(((N_red + x, N_red + y) for x, y in blue_graph.edges()))
    return main_graph


def draw_graph(G : nx.Graph, x : np.ndarray = None):
    '''
    Draws the given graph, optionally coloring the nodes according to the given
    estimate. If the estimate is provided, positive values are colored red, and
    negative values are colored blue.

    Parameters
    ----------
    G : nx.Graph
        The graph to be drawn.
    x : np.ndarray or None
        The estimate for the communities in the graph.
    '''
    c = None
    if x is not None:
        c = [ 'r' if v > 0 else 'b' for v in x ]

    nx.draw(G, node_color=c)
