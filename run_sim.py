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
    fig_name, net_type, task, seed, dev_list=params
    NP, SC = set_params(fig_name=fig_name,
            NET_TYPE=net_type,
            TASK=task,
            SEED=seed,
            DEV_LIST=dev_list)

    sim.createSimulateAnalyze(netParams=NP, simConfig=SC)

    import pylab; pylab.show()  # this line is only necessary in certain systems where figures appear empty

    exit()

if __name__ == '__main__':
    ## basic param modification

    ## grid_search_array - d1, d2, fig_name

    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="enter stimulus task", type=str,
        default='oddball')
    parser.add_argument("type", help="enter type (length)", type=str,
        default='full')
    parser.add_argument("oddball", help="index", type=int,
            default=3)
    args = parser.parse_args()

    TASK = args.task
    TYPE = args.type
    ODDBALL = args.oddball

    SIM_TYPE = TYPE
    SIM_TASK = TASK
    grid_search_array = [
        ['beta_4_{}_{}'.format(TASK,i),
                SIM_TYPE, SIM_TASK, 1, [i]] for i in [ODDBALL] ]


    sim_pool_manager = PoolManager(num_workers=1)
    sim_pool_manager.worker(run_sim, grid_search_array, 999999999)
