"""
Implements the auditory cortex network model from Beeman et al., BMC Neuroscience, 2017 and Kudela et al.,
Frontiers in Neural Circuits, 2018.

"""

from net_params_oddball_sequence import set_params
from netpyne import sim
import numpy as np
from MP_class import PoolManager

from config import D1, D2, LOW_GABA, HIGH_GABA

def run_sim (params):
    # get params
    d1, d2, gaba, name = params
    NP, SC = set_params(d1_param=d1, d2_param=d2, fig_name=name)

    sim.createSimulateAnalyze(netParams=NP, simConfig=SC)

    #import pylab; pylab.show()  # this line is only necessary in certain systems where figures appear empty

    exit()

if __name__ == '__main__':
    ## basic param modification
    timeout_hrs = 600000000

    ## grid_search_array - d1, d2, fig_name
    grid_search_array = [[0.46, 0.76, 'basic_conf']]



    sim_pool_manager = PoolManager(num_workers=4)
    sim_pool_manager.worker(run_sim, grid_search_array, timeout= 60*60*timeout_hrs)
