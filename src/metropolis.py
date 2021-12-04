import numpy as np
import networkx as nx
from scipy import sparse
from performance_utils import generate_population

class Metropolis():
    '''
    Implements an instance of the Metropolis chain.

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix of the graph instance.
    a : float
        Connectivity parameter for nodes in the same community.
    b : float
        Connectivity parameter for nodes in different communities.
    x_star : np.ndarray
        Ground truth vector.
    x_init : np.ndarray or None, optional
        Starting state. If None, the state will be a random vecotr with
        x_init[i] = ±1 chosen uniformly at random.
    '''
    N : int
    A : np.ndarray
    H : np.ndarray
    x : np.ndarray
    n : int

    def __init__(self, A : np.ndarray, a : float, b : float, x_init : np.ndarray = None):
        self.A = A
        self.a = a
        self.b = b
        self.N = A.shape[0]

        # Compute the Hamiltonian
        term1 = A * np.log(a / b)
        term2 = (1. - A) * np.log((1. - a / self. N) / (1. - b / self.N))
        self.H = 0.5 * (term1 + term2)

        self.reset(x_init)


    def __repr__(self):
        return f'Metropolis(N={self.N}, n={self.n})'


    def reset(self, x_init : np.ndarray = None):
        '''
        Resets the Metropolis chain to step 0.

        Parameters
        ----------
        x_init : np.ndarray or None, optional
            Starting state. If None, the state will be a random vecotr with
            x_init[i] = ±1 chosen uniformly at random.
        '''
        self.n = 0

        if x_init is None:
            x_init = generate_population(self.N)
        self.x = x_init.copy()


    def step(self):
        '''
        Moves the Metropolis chain to the next step.
        '''
        # Pick random vertex
        v = np.random.randint(self.N)

        #TODO: there might be more efficient update
        #TODO: update to new np RNG?
        # Compute acceptance probability
        exp = - 2. * self.x[v] * np.inner(self.H[v, :], self.x)
        a_xy = min(1., np.exp(exp))

        # Apply acceptance probability
        sample = np.random.uniform()
        if sample <= a_xy:
            self.x[v] *= -1

        # Update state
        self.n += 1
