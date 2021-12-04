import numpy as np
import networkx as nx
from networkx.algorithms.bipartite import generators
import matplotlib.pyplot as plt


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
    pos = nx.spring_layout(G, seed=2)
    nx.draw(G, pos=pos, node_color=c)
