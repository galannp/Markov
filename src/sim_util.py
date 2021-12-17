import numpy as np
from numpy.lib.function_base import vectorize
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from graph_utils import *
from performance_utils import *
from metropolis import *
from houdayer_algorithm import *
from typing import Iterable


class TimedPrint():
    '''A helper class that allows constant time printing'''
    def __init__(self):
        self.event_time = time.time()

    def next_event_time(self, interval:float):
        self.event_time = time.time() + interval

    def print(self, msg, interval:float=None, end:str='\n', flush=True):
        if time.time() > self.event_time:
            print(msg, end=end, flush=flush)
            self.next_event_time(interval)




def critical_ratio(deg: int) -> float:
    ''' Finds the critical ratio to detect community, one should use r < crit_r.
    
        Parameters:
        ------------
        deg: degree of the graph
        
        Output:
        ------------
        crit_r: critical ratio for successful community detection
    '''
    return (np.sqrt(deg)-1) / (np.sqrt(deg)+1)

def generate_detectable_a_b(deg: int, r: float=None) -> tuple:
    ''' Uniformly generate possible pair (a,b) so that the community is detectable,
        given that (a+b)/2 = deg.

        Parameters:
        -------------
        deg: The degree of the graph
        
        Output: The numbers (a,b,r)
    '''
    rc = critical_ratio(deg)

    if r is None:
        r = np.random.uniform(low=0, high=rc)
    elif r > rc:
        print('Warning: the assigned ratio r = {:} is too large for community detection, need r < {:}.\nChange r value to critical value {:.2f}'.format(r, rc, rc))
        r = rc

    a = 2*deg / (1+r)
    b = a*r

    return a, b, r


def display_spec(N:int, d:int, a:float, b:float, r:float):
    print('Specs:')
    print(f'  Number of nodes (N) = {N}')
    print(f'  Degree of graph (d) = {d}')
    print(f'  Intra-group connect param (a) = {a}')
    print(f'  Inter-group connect param (b) = {b}')
    print(f'  Group connect ratio (r) = b/a = {r}')
    rc = critical_ratio(d)
    print(f'  Critical ratio (rc) = {rc} (for detectable communities)')



def Metropolis_Houdayer_step(G:nx.graph, ch1:Metropolis, ch2:Metropolis, num_metropolis_steps:int=1, num_houdayer_steps:int=1)->None:
    '''
    Apply Houdayer and Metropolis steps to the 2 chains. 
    The result will directly update the estimation stored in the 2 chains.

    Parameters
    -------------
    G: The graph which we want to detect community on.
    ch1, ch2: The 2 chains which we use to sample communities.
    num_metropolis_steps: The number of Metropolis steps we perform
    num_houdayer_steps: The number of Houdayer steps we perform
    '''


    # Update with Houdayer step
    for hou in range(num_houdayer_steps):
        ch1.x, ch2.x = houdayer_step(G, ch1.x, ch2.x)

    # Update with Metropolis step
    for met in range(num_metropolis_steps):
        ch1.step(); ch2.step()



def sim_one_round(G:nx.graph, ch1:Metropolis, ch2:Metropolis, true_group:np.ndarray, num_iter:int, met_steps:int=1, hou_steps:int=1) -> tuple:
    '''
    Run a one time simulation through 2 Metropolis chains.

    Parameter
    ------------
    G: Graph which we try to detect community
    ch1, ch2: The 2 chains we use to sample the true probability
    true_group: The true group label of the nodes
    num_iter: Number of iteration we do the mixed steps.
    met_steps: Number of Metropolis steps in one mixed step.
    hou_steps: Number of Houdayer steps in one mixed step.

    Returns
    ------------
    overlap_ch1: The overlap measurement of the 1st chain (ch1) record for each mixed step
    overlap_ch2: The overlap measurement of the 2nd chain (ch2) record for each mixed step
    '''

    # Obtain the indices (actual number of iteration when the overlap value is sampled)
    iter_index = np.arange(met_steps+hou_steps, num_iter*(met_steps+hou_steps)+1, (met_steps+hou_steps))

    overlap_ch1 = []
    overlap_ch2 = []

    for itr in range(num_iter):
        Metropolis_Houdayer_step(G, ch1, ch2, met_steps, hou_steps)
            
        # Maintain the overlap record
        overlap_ch1.append(compute_overlap(true_group, ch1.x))
        overlap_ch2.append(compute_overlap(true_group, ch2.x))

    return np.array(overlap_ch1), np.array(overlap_ch2), np.array(iter_index)

 

def plot_sim(x:Iterable, y:Iterable, title:str='', x_label:str='', y_label:str='', grid_on:bool=True) -> None:
    '''
    Plot the data x with the assigned x/y labels

    Parameters:
    -------------
    x: a list or array which specify the x-coordinate of data
    y: a list or array of data to be plotted
    x_label: the label of the x-axis
    y_label: the label of the y-axis
    grid_on: whether to turn on the grid or not

    '''

    # Ploting the respective graph
    plt.figure(dpi=600)

    plt.plot(x, y)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(grid_on)



def predict_one(adj:np.ndarray, a:float, b:float, max_iter:int=None, tolerance:float=None, sample_length:int=None) -> np.ndarray:
    '''
    One-round binary community detection by Metropolis-Hasting algorithm.

    Params
    -----------
    adj: the adjacency matrix of graph
    a: intra-group connection parameter
    b: inter-group connection parameter
    max_iter: maximum number of iteration allowed in the simulation
    tolerance: the early termination criteria that if the chain reaches consensus
    sample_length: the length of sample we take to estimate the consensus

    Returns
    -----------
    pred: the predicted state of the graph community
    conv_index: the convergence index with respective the the state output
    '''
    #----------------#
    # Initialisation #
    #----------------#
    ## Initialise the chain and parameters
    chain = Metropolis(adj, a, b)
    num_node = chain.N

    # Assign max_iter as square of number of nodes (complexity of a graph), if it's not defined
    if max_iter is None:
        max_iter = num_node**2
    # Assign sample_length as the number of nodes (expected time to try all nodes), if it's not defiend.
    if sample_length is None:
        sample_length = num_node
    # Assign the tolerance as the inverse of number of nodes (only 1 node is different in average), if it's not defined.
    if tolerance is None:
        tolerance = 1/num_node

    # A helper printing tool to control contant time printing
    time_print = TimedPrint()
    num_digits = int(np.log10(1/tolerance) // 1) +2

    # Initialize the sample states to measure the convergence index,
    # which represents the degree of consensus at the current state of simulation
    recent_states = np.zeros((1, num_node))

    #------------#
    # SIMULATION #
    #------------#

    for itr in range(max_iter):
        # evolve the state and record the current state
        chain.step()
        state = chain.x

        # Maintain the states of the recent 'sample_length' samples to determine degree of consensus
        recent_states = np.vstack([recent_states, state])
        # The criterion to avoid computation when the sample is incomplete
        if itr >= sample_length-1:
            # Maintain the sample window size by discarding old samples
            recent_states = recent_states[1:]

            # The average sample of recent samples
            ave_sample = np.mean(recent_states, axis=0)

            # Current convergence index, the smaller means the recent results reach a consensus
            conv_index = np.mean(np.abs(ave_sample - state)/2)

            # Print current states to keep track of the status quo. Don't let the human operator get boring :-)
            print_msg = "Itr = {itr}; Convergence index = {val:.{num_digits}f}".format(itr=itr+1, val=conv_index, num_digits=num_digits)
            print_time_interval = 0.15  # time interval of each message
            time_print.print(msg=print_msg, interval=print_time_interval, end='\r')

            # If convergence index is sufficiently small (metropolis result close to consensus),
            # terminate the program
            if conv_index < tolerance:
                break

    # Make sure to print the last iteration when terminating
    print(print_msg)

    # Get the first majority vote from the recent states
    pred = get_majority(recent_states)
    
    return pred, conv_index

    

def predict(adj:np.ndarray, a:float, b:float, true_label:np.ndarray=None,
                     sim_num:int=1, max_iter:int=None, tolerance:float=None, sample_length:int=None):
    '''
    Perform multiple simulations to determine the desired output by weighted majority vote.
    weights are the inverse of convergence indices according to each simulations

    Params
    ---------
    adj: the adjacency matrix of graph
    a: intra-group connection parameter
    b: inter-group connection parameter
    true_label: (optional) the true label of the community. Only for printing purpose, doesn't affect the result.
    sim_num: number of simulations to perform
    max_iter: maximum number of iteration allowed in the simulation
    tolerance: the early termination criteria that if the chain reaches consensus
    sample_length: the length of sample we take to estimate the consensus

    Returns
    ---------
    pred_major: the predicted labels produced by the weighted majority vote of the each simulation
    '''
    # Initialize the container for the output of each simulation
    # predictions: each row is the prediction at that simulation round
    predictions = np.zeros((sim_num, adj.shape[0]))
    # each element is the convergence index at that simulation round
    conv_indices = np.ones(sim_num)

    # Simulate for the given times
    for s in range(sim_num):
        print(f"Sim - {s+1}")
        predictions[s,:] ,conv_indices[s] = predict_one(adj, a, b, max_iter, tolerance, sample_length)
        if true_label is not None:
            print(f"Overlap = {compute_overlap(true_label, predictions[s,:])}")

    # Get the weighted majority vote from all the simulations
    pred_major = get_majority(predictions, 1/conv_indices)

    return pred_major


def get_majority(samples, weight=None):
    '''
    Get the weighted/unweighted majority vote of all the samples given.

    Params
    ----------
    samples: S-by-N matrix where S is the number of simulations and N is the number of nodes
    weight: the weights to give to each simulation

    Returns
    ----------
    major: the prediction by the majority vote
    '''
    ## Direction tilting: 
    # We rotate each prediction to the same direction. Since prediction x and -x represent
    # the same community calssification. But when we average all the results, it causes problems.

    # row-wise tilting the vectors to the same direction as the first sample
    vec_invert = np.vectorize(same_direction)
    # Turning all the vectors to the same direction
    samples = vec_invert(samples, samples[0])

    # Simple majority is the sign of the sum
    if weight is None:
        major = np.sum(samples, axis=0)
    else:
        # Normalize the weight to ensure correct answer
        w = weight / np.sum(weight)
        major = np.average(samples, axis=0, weights=w)

    major = np.sign(major)
    # Handles exception that 0 happens, we just randomly assign one group to it.
    major[major == 0] = 1 if np.random.rand() > 0.5 else -1

    return major


def same_direction(vec, ref):
    '''
    Turn vec to the same direction as ref
    '''
    if np.dot(vec, ref) < 0:
        return -vec
    else:
        return vec