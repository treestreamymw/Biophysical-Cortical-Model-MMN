"""
Implements the auditory cortex network model from Beeman et al., BMC Neuroscience, 2017 and Kudela et al.,
Frontiers in Neural Circuits, 2018.

"""
from mpi4py import MPI
from netpyne import sim
from net_params import set_params
import numpy as np
from MP_class import PoolManager

def run_sim (params):
    # get params
    fig_name, net_type, task, debug_params, dev_list=params
    NP, SC = set_params(fig_name=fig_name,
            NET_TYPE=net_type,
            TASK=task,
            DEBUG_PARAMS=debug_params,
            DEV_LIST=dev_list)

    sim.createSimulateAnalyze(netParams=NP, simConfig=SC)

    import pylab; pylab.show()  # this line is only necessary in certain systems where figures appear empty

    exit()

if __name__ == '__main__':
    ## basic param modification

    ## grid_search_array - d1, d2, fig_name
    SIM_TYPE='short' #short
    TASK='flipflop'
    grid_search_array = [
        ['beta_network_2_{}'.format(i),
                SIM_TYPE, TASK, 1, [i]] for i in [2] ]


    sim_pool_manager = PoolManager(num_workers=1)
    sim_pool_manager.worker(run_sim, grid_search_array, 999999999)
