
import os
import networkx as nx
import numpy as np

from src.sim_util import *

# Get the path to load the adjacency matrix of the task
CWD = os.getcwd()
ADJ_MAT_PATH = os.path.join(CWD, 'A_test.npy')

OUTPUT_NAME = 'prediction.npy'
OUTPUT_PATH = os.path.join(CWD, )

'''Define parameters'''
#-------------#
# task params #
#-------------#
# Connection parameter a & b
a = ...
b = ...
# Construct graph by the given adjacency matrix
adjacency_mat = np.load(ADJ_MAT_PATH)
graph = nx.from_numpy_array(adjacency_mat)

#-------------------#
# simulation params #
#-------------------#
sim_num = 25     # Number of simulations
max_iter = None  # Max number of iterations for each simulation. Can be left to None, default value = (number of nodes)^2
tolerance = None # Early termination criterion for each simulation. Can be left to None, default value = 1/(number of nodes)
samp_leng = None # Sample length of convergence evaluation for deciding early termination. Can be left to None, default value = (number of nodes)

'''Simulation'''
prediction = predict(adjacency_mat, a, b,
                     sim_num=sim_num,
                     max_iter=max_iter,
                     tolerance=tolerance,
                     sample_length=samp_leng)


# Save the output file
np.save(OUTPUT_PATH, prediction)