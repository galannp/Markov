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
    nx.draw(G, pos=pos, node_color=c, node_size=12, width=0.5)



def graph_gen(group:np.ndarray, a:int, b:int,) -> (nx.graph):
    ''' 
    Generate the graph using the ground truth and the assigned parameters a and b

    Parameters
    --------------
    group: list of +1, -1 to indicate the group belonging relations
    a: integer to define the proability of connection as a/N given that both vertices are within the same group. a < N
    b: integer to define the proability of connection as b/N given that both vertices are in different groups. b < N

    Returns
    --------------
    graph: an networkx graph object

    Example:
    >>> np.random.seed(0)
    >>> x = np.array([1, -1, 1, 1, -1])
    >>> G = graph_gen(x, 4, 1)
    >>> print(G)
    >>> 
    [[0 0 1 1 0]
     [0 0 0 1 1]
     [1 0 0 0 0]
     [1 1 0 0 0]
     [0 1 0 0 0]]
    '''

    # determine graph size 
    N = group.size

    # edge type (+1 = same group and -1 = different)
    edge_type = np.outer(group, group)
    # Boolean matrices to indicate same group and different group
    same_group = edge_type > 0
    diff_group = edge_type < 0

    # create a random matrix to generate the graph
    sample = np.random.rand(N,N)
    # define the probability of existing an edge
    connect_prob_same = a / N
    connect_prob_diff = b / N

    # sample the edges which should exist according to their respective probability
    pass_edges_same = np.logical_and(sample <= connect_prob_same, same_group)
    pass_edges_diff = np.logical_and(sample <= connect_prob_diff, diff_group)

    A = np.logical_or(pass_edges_same, pass_edges_diff).astype(int)

    # Sample the lower triangular matrix to make the matrix symmetric (the graph is undirected)
    Adj = np.tril(A, -1)
    Adj = Adj + Adj.T

    # Transform the adjacency matrix into a graph object
    graph = nx.from_numpy_array(Adj)

    return graph
