import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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


def generate_adjacency(x_star : np.ndarray, a : float, b : float, seed : int = None) -> np.ndarray:
    '''
    Generates the adjacency matrix of a random graph from the ground truth and
    connectivity parameters.

    Parameters
    ----------
    x_star : np.ndarray
        Ground truth vector.
    a : float
        Connectivity parameter for nodes in the same community.
    b : float
        Connectivity parameter for nodes in different communities.
    seed : int or None
        The random seed. Can be None to use a built-in source of randomness.

    Returns
    -------
    A : np.ndarray
        The generated adjacency matrix.
    '''
    np.random.seed(seed)

    # Build adjacency matrix A
    N = len(x_star)
    X = np.outer(x_star, x_star)
    T = np.where(X == 1, a/N, b/N)
    R = np.random.uniform(size=X.shape) + np.identity(N)
    A = (R < T).astype(int)

    return A


def build_graph(A : np.ndarray) -> nx.Graph:
    '''
    Constructs a networkx graph from an adjacency matrix.

    Parameters
    ----------
    A : np.ndarray
        The adjacency matrix.

    Returns
    -------
    G : nx.Graph
        The graph corresponding to A.
    '''
    return nx.convert_matrix.from_numpy_matrix(A)


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
