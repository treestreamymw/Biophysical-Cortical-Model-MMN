"""
Implements the auditory cortex network model from Beeman et al., BMC Neuroscience, 2017 and Kudela et al.,
Frontiers in Neural Circuits, 2018.

"""
from mpi4py import MPI
from netpyne import sim
from net_params import set_params
import argparse

#from net_params_SSA import set_params

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

    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="enter stimulus task", type=str)
    parser.add_argument("type", help="enter type (length)", type=str)
    args = parser.parse_args()

    TASK=args.task
    TYPE=args.type

    SIM_TYPE = TYPE or 'full'#'full' #short
    SIM_TASK = TASK or 'oddball'
    grid_search_array = [
        ['beta_4_many_standards_{}'.format(i),
                SIM_TYPE, SIM_TASK, 1, [i]] for i in [5] ]


    sim_pool_manager = PoolManager(num_workers=1)
    sim_pool_manager.worker(run_sim, grid_search_array, 999999999)
