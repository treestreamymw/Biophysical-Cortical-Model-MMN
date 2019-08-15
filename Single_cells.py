from Single_cells_params import set_params
from netpyne import sim
import numpy as np
from MP_class import PoolManager

def run_sim (params):
    # get params

    NP, SC = set_params(input_rs_threshold=params)

    sim.createSimulateAnalyze(netParams=NP, simConfig=SC)

    import pylab; pylab.show()  # this line is only necessary in certain systems where figures appear empty

    exit()

if __name__ == '__main__':
    ## basic param modification

    grid_search_array = [0.04]


    sim_pool_manager = PoolManager(num_workers=10)
    sim_pool_manager.worker(run_sim, grid_search_array, 999999999)
