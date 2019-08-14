"""
Single_cells_params.py

netParams is a dict containing a set of network parameters using a standardized structure

simConfig is a dict containing a set of simulation configurations using a standardized structure

Contributors: Christoph Metzner, christoph.metzner@gmail.com, 05/08/2019
"""

from netpyne import specs
import numpy as np

netParams = specs.NetParams()   # object of class NetParams to store the network parameters
simConfig = specs.SimConfig()   # object of class SimConfig to store the simulation configuration

###############################################################################
# NETWORK PARAMETERS
###############################################################################

# Population parameters
netParams.popParams['PYR'] = {'cellModel': 'PYR', 'cellType': 'PYR', 'numCells': 1,  'color': 'blue'}


# Cell parameters
## PYR cell properties
cellRule = netParams.importCellParams(label='PYR',
	conds= {'cellType': 'PYR', 'cellModel': 'PYR'},
	fileName='Cells/pyr_23_asym_stripped.hoc',
	cellName='pyr_23_asym_stripped')


# ToDo: Change!
currents_rs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
threshold_rs = 0.07


for i,c in enumerate(currents_rs):
	current = c + threshold_rs
	delay = 200 + i*2000
	netParams.stimSourceParams['ic'+str(i)] = {'type': 'IClamp',
			'delay': delay, 'dur': 1000.0, 'amp': current}
	netParams.stimTargetParams['ic->PYR'+str(i)] = {'source': 'ic'+str(i),
				'conds': {'popLabel': 'PYR'},
				'sec':'soma','loc':0.5}



###############################################################################
# SIMULATION PARAMETERS
###############################################################################

# Simulation parameters
simConfig.hParams['celsius'] = 30.0
simConfig.duration = 10000 # Duration of the simulation, in ms
simConfig.dt = 0.025 # Internal integration timestep to use
simConfig.seeds = {'conn': 1, 'stim': 1, 'loc': 1} # Seeds for randomizers (connectivity, input stimulation and cell locations)
simConfig.createNEURONObj = 1  # create HOC objects when instantiating network
simConfig.createPyStruct = 1  # create Python structure (simulator-independent) when instantiating network
simConfig.verbose = False  # show detailed messages
simConfig.printPopAvgRates = True
#simConfig.verbose = True

# Recording
simConfig.recordCells = ['all']  # which cells to record from
simConfig.recordStep = 0.1 # Step size in ms to save data (eg. V traces, LFP, etc)
simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}



# Saving
simConfig.filename = 'test'  # Set file output name
simConfig.saveFileStep = 1000 # step size in ms to save data to disk
simConfig.saveJson = True # Whether or not to write spikes etc. to a .json file

simConfig.analysis['plotTraces'] = {'include': [0]}
