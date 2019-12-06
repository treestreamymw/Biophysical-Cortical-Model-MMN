"""
net_params.py

netParams is a dict containing a set of network parameters using a standardized structure

simConfig is a dict containing a set of simulation configurations using a standardized structure

Implements the auditory cortex network model from Beeman et al., BMC Neuroscience, 2017 and Kudela et al.,
Frontiers in Neural Circuits, 2018. Cell types have been changed!

Contributor: Christoph Metzner, christoph.metzner@gmail.com, 17/09/2018
"""

from netpyne import specs

def set_params(fig_name, NET_TYPE, TASK, DEBUG_PARAMS, DEV_LIST):

    netParams = specs.NetParams()   # object of class NetParams to store the network parameters
    simConfig = specs.SimConfig()   # object of class SimConfig to store the simulation configuration

    ###############################################################################
    # NETWORK PARAMETERS
    ###############################################################################

    netParams.sizeX = 1880
    netParams.sizeY = 2500
    netParams.sizeZ = 1880

    # Population parameters
    netParams.popParams['PYR23'] = {'cellModel': 'PYR_Hay', 'cellType': 'PYR', 'gridSpacing': 40.0, 'yRange': [2000, 2000],
                                    'color': 'blue'}
    netParams.popParams['PYR4'] = {'cellModel': 'PYR_Hay', 'cellType': 'PYR', 'gridSpacing': 40.0, 'yRange': [2500, 2500],
                                   'color': 'green'}
    netParams.popParams['BASK23'] = {'cellModel': 'BASK_Vierling', 'cellType': 'BASK', 'gridSpacing': 80.0,
                                     'xRange': [20, 1880], 'zRange': [20, 1880], 'yRange': [2000, 2000], 'color': 'red'}
    netParams.popParams['BASK4'] = {'cellModel': 'BASK_Vierling', 'cellType': 'BASK', 'gridSpacing': 80.0,
                                    'xRange': [20, 1880], 'zRange': [20, 1880], 'yRange': [2500, 2500], 'color': 'yellow'}

    # Cell parameters

    # PYR cell properties
    cellRule = netParams.importCellParams(label='PYR', conds={'cellType': 'PYR', 'cellModel': 'PYR_Hay'},
                                          fileName='Cells/fourcompartment.hoc', cellName='fourcompartment')
    cellRule['secs']['soma']['vinit'] = -80.0
    cellRule['secs']['dend']['vinit'] = -80.0
    cellRule['secs']['apic_0']['vinit'] = -80.0
    cellRule['secs']['apic_1']['vinit'] = -80.0

    # INH cell properties
    cellRule = netParams.importCellParams(label='BASK', conds={'cellType': 'BASK', 'cellModel': 'BASK_Vierling'},
                                          fileName='Cells/FS.hoc', cellName='Layer2_basket')

    # Synaptic mechanism parameters
    netParams.synMechParams['AMPA'] = {'mod': 'Exp2Syn', 'tau1': 1.0, 'tau2': 3.0, 'e': 0.0}
    netParams.synMechParams['AMPASTD'] = {'mod': 'FDSExp2Syn', 'tau1': 1.0, 'tau2': 3.0, 'e': 0.0, 'f': 0.0, 'tau_F': 94.0,
                                          'd1': .46, 'tau_D1': 380, 'd2': .76, 'tau_D2': 9200}  # only depression
    netParams.synMechParams['GABA'] = {'mod': 'Exp2Syn', 'tau1': 5.0, 'tau2': 12.0, 'e': -75.0}


    # Stimulation parameters
    #  netParams.stimSourceParams['bkg'] = {'type': 'NetStim','rate': 40.0,'noise': 1.0, 'start': 0.0}  # background noise
    #  netParams.stimTargetParams['bkg->PYR23'] = {'source': 'bkg', 'conds': {'popLabel': 'PYR23'}, 'weight': 0.0075}
    #  netParams.stimTargetParams['bkg->PYR4'] = {'source': 'bkg', 'conds': {'popLabel': 'PYR4'}, 'weight': 0.0075}
    #  netParams.stimTargetParams['bkg->BASK23'] = {'source': 'bkg', 'conds': {'popLabel': 'BASK23'}, 'weight': 0.0005}
    #  netParams.stimTargetParams['bkg->BASK4'] = {'source': 'bkg', 'conds': {'popLabel': 'BASK4'}, 'weight': 0.0005}


    pulse1 = [{'start': 500.0, 'end': 700, 'rate': 200, 'noise': 1.0}]
    pulse2 = [{'start': 1500.0, 'end': 1700, 'rate': 200, 'noise': 1.0}]

    netParams.popParams['Stim1000Hz'] = {'cellModel': 'VecStim', 'numCells': 48, 'spkTimes': [0], 'pulses': pulse1 }
    netParams.connParams['Stim1000Hz->PYR4'] = {
        'preConds': {'popLabel': 'Stim1000Hz'}, 'postConds': {'popLabel': 'PYR4', 'x': [679, 681]},
        'convergence': 1,
        'weight': 0.02,
        'threshold': 10,
        'synMech': 'AMPA'}

    netParams.connParams['Stim1000Hz->BASK4'] = {
        'preConds': {'popLabel': 'Stim1000Hz'}, 'postConds': {'popLabel': 'BASK4', 'x': [659, 661]},
        'convergence': 1,
        'weight': 0.02,
        'threshold': 10,
        'synMech': 'AMPA'}

    netParams.popParams['Stim1200Hz'] = {'cellModel': 'VecStim', 'numCells': 48, 'spkTimes': [0], 'pulses': pulse2 }
    netParams.connParams['Stim1200Hz->PYR4'] = {
        'preConds': {'popLabel': 'Stim1200Hz'}, 'postConds': {'popLabel': 'PYR4', 'x': [1159, 1161]},
        'convergence': 1,
        'weight': 0.02,
        'threshold': 10,
        'synMech': 'AMPA'}

    netParams.connParams['Stim1200Hz->BASK4'] = {
        'preConds': {'popLabel': 'Stim1200Hz'}, 'postConds': {'popLabel': 'BASK4', 'x': [1139, 1141]},
        'convergence': 1,
        'weight': 0.02,
        'threshold': 10,
        'synMech': 'AMPA'}

    # Connectivity parameters

    # Layer 4 intra-laminar connections
    netParams.connParams['PYR4->PYR4'] = {
        'preConds': {'popLabel': 'PYR4'}, 'postConds': {'popLabel': 'PYR4'},
        'sec': 'apic_1',
        'probability': '0.15*exp(-dist_3D/(4*40.0))',
        'weight': 0.004,
        'threshold': 10,
        'synMech': 'AMPA'}

    netParams.connParams['PYR4->BASK4'] = {
        'preConds': {'popLabel': 'PYR4'}, 'postConds': {'popLabel': 'BASK4'},
        'probability': '0.45*exp(-dist_3D/(4*40.0))',
        'weight': 0.002,
        'threshold': 10,
        'synMech': 'AMPA'}

    netParams.connParams['BASK4->PYR4'] = {
        'preConds': {'popLabel': 'BASK4'}, 'postConds': {'popLabel': 'PYR4'},
        'probability': '0.6*exp(-dist_3D/(4*40.0))',
        'weight': 0.003,
        'threshold': 10,
        'synMech': 'GABA'}
    '''
    netParams.connParams['BASK4->BASK4'] = {
        'preConds': {'popLabel': 'BASK4'}, 'postConds': {'popLabel': 'BASK4'},
        'probability': '0.6*exp(-dist_3D/(4*40.0))',
        'weight': 0.0,
        'threshold': 10,
        'synMech': 'GABA'}
    '''

    # Layer 2/3 intra-laminar connections
    netParams.connParams['PYR23->PYR23'] = {
        'preConds': {'popLabel': 'PYR23'}, 'postConds': {'popLabel': 'PYR23'},
        'sec': 'apic_1',
        'probability': '0.15*exp(-dist_3D/(4*40.0))',
        'weight': 0.006,
        'threshold': 10,
        'synMech': 'AMPA'}

    netParams.connParams['PYR23->BASK23'] = {
        'preConds': {'popLabel': 'PYR23'}, 'postConds': {'popLabel': 'BASK23'},
        'probability': '0.45*exp(-dist_3D/(4*40.0))',
        'weight': 0.002,
        'threshold': 10,
        'synMech': 'AMPA'}

    netParams.connParams['BASK23->PYR23'] = {
        'preConds': {'popLabel': 'BASK23'}, 'postConds': {'popLabel': 'PYR23'},
        'probability': '0.6*exp(-dist_3D/(4*40.0))',
        'weight': 0.002,
        'threshold': 10,
        'synMech': 'GABA'}
    '''
    netParams.connParams['BASK23->BASK23'] = {
        'preConds': {'popLabel': 'BASK23'}, 'postConds': {'popLabel': 'BASK23'},
        'probability': '0.6*exp(-dist_3D/(4*40.0))',
        'weight': 0.0,
        'threshold': 10,
        'synMech': 'GABA'}
    '''
    # Inter-laminar connections

    netParams.connParams['PYR4->PYR23'] = {
        'preConds': {'popLabel': 'PYR4'}, 'postConds': {'popLabel': 'PYR23'},
        'sec': 'dend',
        'probability': '0.5*exp(-dist_2D/(2*40.0))',
        'weight': 0.03,
        'threshold': 10,
        'synMech': 'AMPASTD'}


    netParams.connParams['PYR4->BASK23'] = {
        'preConds': {'popLabel': 'PYR4'}, 'postConds': {'popLabel': 'BASK23'},
        'probability': '0.8*exp(-dist_2D/(2*40.0))',
        'weight': 0.00015,
        'threshold': 10,
        'synMech': 'AMPASTD'}

    ###############################################################################
    # SIMULATION PARAMETERS
    ###############################################################################

    # Simulation parameters
    simConfig.hParams['celsius'] = 30.0
    simConfig.duration = 2000  # Duration of the simulation, in ms
    simConfig.dt = 0.05  # Internal integration timestep to use
    simConfig.seeds = {'conn': 1, 'stim': 1, 'loc': 1}  # Seeds for randomizers (conn., input stim. and cell loc.)
    simConfig.createNEURONObj = 1  # create HOC objects when instantiating network
    simConfig.createPyStruct = 1  # create Python structure (simulator-independent) when instantiating network
    simConfig.verbose = False  # show detailed messages
    simConfig.hParams['cai0_ca_ion'] = 0.0001
    simConfig.printPopAvgRates = True


    # Recording
    simConfig.recordCells = ['all']  # which cells to record from
    simConfig.recordTraces = {'Vsoma': {'sec': 'soma', 'loc': 0.5, 'var': 'v'}}
    # ,'AMPA':{'sec':'dend','loc':0.5,'var':'AMPA','conds':{'cellType':'PYR'}}}
    simConfig.recordStim = True  # record spikes of cell stims
    simConfig.recordStep = 0.1  # Step size in ms to save data (eg. V traces, LFP, etc)
    simConfig.recordLFP = [[680, 0, 990], [1160, 0, 990]]  # electrodes at 1000Hz and 1200Hz

    # Saving
    simConfig.filename = 'test'  # Set file output name
    simConfig.saveFileStep = 1000  # step size in ms to save data to disk
    simConfig.savePickle = False  # True  # Whether or not to write spikes etc. to a .pkl file

    # Analysis and plotting
    simConfig.analysis['plotRaster'] = {'saveFig': '{}_raster.png'.format(fig_name)}  # Plot raster
    #simConfig.analysis['plotTraces'] = {'include': [5567, 5568, 5569], 'saveFig': True}  # Plot raster
    # simConfig.analysis['plot2Dnet'] = {'view': 'xz','showConns': False}  # Plot 2D net cells and connections
    #simConfig.analysis['plot2Dnet'] = {'include': [('PYR4', 1000), 'PYR23'], 'view': 'xy', 'showConns': True}  # Plot 2D net cells and connections
    simConfig.analysis['plotLFP'] = {'includeAxon': False, 'plots': ['timeSeries'], 'figSize': (5, 9), 'saveFig':  '{}_LFP.png'.format(fig_name)}


    return (netParams, simConfig)
