"""
net_params.py

netParams is a dict containing a set of network parameters using a standardized structure

simConfig is a dict containing a set of simulation configurations using a standardized structure

"""
import numpy as np
from mpi4py import MPI
from netpyne import specs
from config import SIM_PARAMS
from stimulus_utils import Simulation_stimuli_Handler

def set_params(fig_name, NET_TYPE, TASK, DEBUG_PARAMS):
    p= DEBUG_PARAMS
    netParams = specs.NetParams()   # object of class NetParams to store the network parameters
    simConfig = specs.SimConfig()   # object of class SimConfig to store the simulation configuration

    ###############################################################################
    # NETWORK PARAMETERS
    ###############################################################################

    # Size and scale
    netParams.sizeX, netParams.sizeY, netParams.sizeZ=SIM_PARAMS[NET_TYPE]['size']
    netParams.scaleConnWeight = SIM_PARAMS[NET_TYPE]['scale']
    netParams.scaleConnWeightNetStims = SIM_PARAMS[NET_TYPE]['scale']

    # Population parameters

    ## excitatory cells layer 2/3
    netParams.popParams['PYR23'] = {'cellModel': 'PYR_Hay', 'cellType': 'PYR',
                                    'gridSpacing': 40.0,
                                    'xRange': [20, netParams.sizeX],
                                    'yRange': [.8*netParams.sizeY,
                                                .8*netParams.sizeY],
                                    'color': 'blue'}

    ## secondary excitatory cells layer 2/3
    netParams.popParams['PYR23_2nd'] = {'cellModel': 'PYR_Hay', 'cellType': 'PYR',
                                    'gridSpacing': 40.0,
                                    'xRange': [20, netParams.sizeX],
                                    'yRange': [.6*netParams.sizeY,
                                                .6*netParams.sizeY],
                                    'color': 'purple'}
    ## excitatory cells layer 4
    netParams.popParams['PYR4'] = {'cellModel': 'PYR_Hay', 'cellType': 'PYR',
                                   'gridSpacing': 40.0,
                                   'xRange': [20, netParams.sizeX],
                                   'yRange': [netParams.sizeY,
                                                netParams.sizeY],
                                   'color': 'green'}
    ## inhibitory cells layer 2/3
    netParams.popParams['BASK23'] = {'cellModel': 'BASK_Vierling',
                                     'cellType': 'BASK', 'gridSpacing': 80.0,
                                     'yRange': [.8*netParams.sizeY,
                                                .8*netParams.sizeY],
                                     'color': 'red'}

    ## inhibitory cells layer 4
    netParams.popParams['BASK4'] = {'cellModel': 'BASK_Vierling',
                                    'cellType': 'BASK', 'gridSpacing': 80.0,
                                    'yRange': [netParams.sizeY,
                                                netParams.sizeY],
                                    'color': 'yellow'}

    # Cell parameters

    # PYR cell properties ()
    cellRule = netParams.importCellParams(label='PYR4', conds={'cellType': 'PYR',
                                                       'cellModel': 'PYR_Hay'},
                                          fileName='Cells/fourcompartment.hoc',
                                  		cellName='fourcompartment')

    cellRule['secs']['soma']['vinit'] = -80.0
    cellRule['secs']['dend']['vinit'] = -80.0
    cellRule['secs']['apic_0']['vinit'] = -80.0
    cellRule['secs']['apic_1']['vinit'] = -80.0

    cellRule = netParams.importCellParams(label='PYR23', conds={'cellType': 'PYR',
                                                       'cellModel': 'PYR_Hay'},
                                          fileName='Cells/pyr_23_asym_stripped.hoc',
                                  		cellName='pyr_23_asym_stripped')

    cellRule = netParams.importCellParams(label='PYR23_2nd', conds={'cellType': 'PYR',
                                                       'cellModel': 'PYR_Hay'},
                                          fileName='Cells/pyr_23_asym_stripped.hoc',
                                  		cellName='pyr_23_asym_stripped')

    # BASK cell properties (INH)
    cellRule = netParams.importCellParams(label='BASK', conds={'cellType': 'BASK',
                                                  'cellModel': 'BASK_Vierling'},
                                          fileName='Cells/FS.hoc',
                                          cellName='Layer2_basket')

    # Synaptic mechanism parameters
    netParams.synMechParams['AMPA'] = {'mod': 'Exp2Syn', 'tau1': 1.0,
                                       'tau2': 3.0, 'e': 0.0}
    netParams.synMechParams['AMPASTD'] = {'mod': 'FDSExp2Syn', 'tau1': 1.0,
                                          'tau2': 3.0, 'e': 0.0, 'f': 0.0,
                                          'tau_F': 94.0, 'd1': 0.46,
                                          'tau_D1': 380, 'd2': 0.76,
                                          'tau_D2': 9200}  # only depression
    netParams.synMechParams['GABA'] = {'mod': 'Exp2Syn',
                                       'tau1': 5.0, 'tau2': 12.0, 'e': -75.0}
    netParams.synMechParams['NMDA'] = {'mod': 'NMDA', 'Alpha':10.0,
                                      'Beta':0.015, 'e':45.0,'g':1,'gmax':1}


    ###############################################################################
    # STIMULI PARAMETERS
    ###############################################################################


    deviant_pulses_indexes = np.random.choice(range(SIM_PARAMS[NET_TYPE]['n_pulses']),
            SIM_PARAMS[NET_TYPE]['n_dev'], replace=False)

    s_handler = Simulation_stimuli_Handler(net_x_size=netParams.sizeX,
                                n_pulses=SIM_PARAMS[NET_TYPE]['n_pulses'],
                                spacing=40.0,
                                dev_indexes=[SIM_PARAMS[NET_TYPE]['n_pulses']-1],
                                task=TASK)
    s_handler.set_task_stimuli()

    external_pulses_info=s_handler.get_formatted_pulse(external=True)
    external_pulses_time=s_handler.get_pulse_time(external=True)

    internal_pulses_info=s_handler.get_formatted_pulse(external=False)
    internal_pulses_time=s_handler.get_pulse_time(external=False)

    for t_pulse in external_pulses_info.keys():

        # External sensory stimuli
        ext_stim_pop_name='Stim_' + str(external_pulses_info[t_pulse]['pop_name']) +"_"+ str(t_pulse)

        netParams.popParams[ext_stim_pop_name] =  {'cellModel': 'VecStim',
                   'numCells': 24, 'spkTimes':[0],
                   'pulses':[{'start': t_pulse*1000+external_pulses_time[0],
                       'end': t_pulse*1000.0+external_pulses_time[1], 'rate': 200,
                       'noise': 1.0}]}


        ext_x_pyr,ext_x_bask=external_pulses_info[t_pulse]['values']

        netParams.connParams[ext_stim_pop_name + '->PYR4'] = {
            'preConds': {'popLabel': ext_stim_pop_name},
            'postConds': {'popLabel': 'PYR4', 'x': ext_x_pyr},
            'convergence': 1,
            'weight': 0.02,
            'threshold': 10,
            'synMech': 'AMPA'}

        netParams.connParams[ext_stim_pop_name + '->BASK4'] = {
            'preConds': {'popLabel': ext_stim_pop_name},
            'postConds': {'popLabel': 'BASK4', 'x': ext_x_bask},
            'convergence': 1,
            'weight': 0.02,
            'threshold': 10,
            'synMech': 'AMPA'}

        # Internal stim - memory trace
        '''
        int_stim_pop_name='internal_' + str(internal_pulses_info[t_pulse]['pop_name']) +"_"+ str(t_pulse)

        netParams.popParams[int_stim_pop_name] =  {'cellModel': 'VecStim',
                       'numCells': 24, 'spkTimes':[0],
                       'pulses':[{'start': t_pulse*1000+internal_pulses_time[0],
                           'end': t_pulse*1000.0+internal_pulses_time[1], 'rate': 200,
                           'noise': 1.0}]}
        int_x_pyr,int_x_bask=internal_pulses_info[t_pulse]['values']

        netParams.connParams[ext_stim_pop_name + '->PYR23_2nd'] = {
            'preConds': {'popLabel': int_stim_pop_name},
            'postConds': {'popLabel': 'PYR23_2nd', 'x': int_x_pyr},
            'convergence': 1,
            'weight': 0.02,
            'threshold': 10,
            'synMech': 'AMPA'}

        '''
    ###############################################################################
    # CONNECTIVITY PARAMETERS
    ###############################################################################

    # Layer 4 intra-laminar connections
    netParams.connParams['PYR4->PYR4'] = {
        'preConds': {'popLabel': 'PYR4'}, 'postConds': {'popLabel': 'PYR4'},
        'sec': 'apic_1',
        'probability': '0.15*exp(-dist_3D/(4*40.0))',
        'weight': [0.0012,0.0006],
        'threshold': 10,
        'synMech': ['AMPA','NMDA']}

    netParams.connParams['PYR4->BASK4'] = {
        'preConds': {'popLabel': 'PYR4'}, 'postConds': {'popLabel': 'BASK4'},
        'probability': '0.45*exp(-dist_3D/(4*40.0))',
        'weight': [0.0012,0.00013],
        'threshold': 10,
        'synMech': ['AMPA','NMDA']}

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
        'sec':'oblique2b',
        'probability': '0.15*exp(-dist_3D/(4*40.0))',
        'weight': [0.0024,0.00012],#[0.0012,0.0006],
        'threshold': 10,
        'synMech': ['AMPA','NMDA']}

    '''netParams.connParams['PYR23_2nd->PYR23'] = {
        'preConds': {'popLabel': 'PYR23_2nd'}, 'postConds': {'popLabel': 'PYR23'},
        'sec':'oblique2b',
        'probability': '0.15*exp(-dist_3D/(4*40.0))',
        'weight': [0.0024,0.00012],#[0.0012,0.0006],
        'threshold': 10,
        'synMech': ['AMPA','NMDA']}'''

    '''netParams.connParams['PYR23->PYR23_2nd'] = {
        'preConds': {'popLabel': 'PYR23'}, 'postConds': {'popLabel': 'PYR23_2nd'},
        'sec':'oblique2b',
        'probability': '0.15*exp(-dist_3D/(4*40.0))',
        'weight': [0.0024,0.00012],#[0.0012,0.0006],
        'threshold': 10,
        'synMech': ['AMPA','NMDA']}'''

    netParams.connParams['PYR23->BASK23'] = {
        'preConds': {'popLabel': 'PYR23'}, 'postConds': {'popLabel': 'BASK23'},
        'probability': '0.45*exp(-dist_3D/(4*40.0))',
        'weight': [0.0012,0.00013],
        'threshold': 10,
        'synMech': ['AMPA','NMDA']}

    '''netParams.connParams['PYR23_2nd->BASK23'] = {
        'preConds': {'popLabel': 'PYR23_2nd'}, 'postConds': {'popLabel': 'BASK23'},
        'probability': '0.45*exp(-dist_3D/(4*40.0))',
        'weight': [0.0012,0.00013],
        'threshold': 10,
        'synMech': ['AMPA','NMDA']}'''

    netParams.connParams['BASK23->PYR23'] = {
        'preConds': {'popLabel': 'BASK23'}, 'postConds': {'popLabel': 'PYR23'},
        'sec':'oblique2a',
        'probability': '0.6*exp(-dist_3D/(4*40.0))',
        'weight': 0.002,
        'threshold': 10,
        'synMech': 'GABA'}


    '''netParams.connParams['BASK23->BASK23'] = {
        'preConds': {'popLabel': 'BASK23'}, 'postConds': {'popLabel': 'BASK23'},
        'probability': '0.6*exp(-dist_3D/(4*40.0))',
        'weight': 0.0,
        'threshold': 10,
        'synMech': 'GABA'}'''


    ### inter-laminar connections
    netParams.connParams['PYR4->PYR23'] = {
        'preConds': {'popLabel': 'PYR4'}, 'postConds': {'popLabel': 'PYR23'},
        'sec': 'basal2b',
        'probability': '0.5*exp(-dist_2D/(2*40.0))',
        'weight': 0.03,
        'threshold': 10,
        'synMech': 'AMPASTD'}


    netParams.connParams['PYR4->BASK23'] = {
        'preConds': {'popLabel': 'PYR4'}, 'postConds': {'popLabel': 'BASK23'},
        'probability': '0.8*exp(-dist_2D/(4*40.0))',
        'weight': 0.0015,
        'threshold': 10,
        'synMech': 'AMPASTD'}


    ########################################################################
    # SIMULATION PARAMETERS
    ########################################################################

    # Simulation parameters
    simConfig.hParams['celsius'] = 30.0
    simConfig.duration = SIM_PARAMS[NET_TYPE]['duration']  # Duration of the simulation, in ms
    simConfig.dt = 0.05  # Internal integration timestep to use
    simConfig.seeds = {'conn': 1, 'stim': 1, 'loc': 1}  # Seeds for randomizers (conn., input stim. and cell loc.)
    simConfig.createNEURONObj = 1  # create HOC objects when instantiating network
    simConfig.createPyStruct = 1  # create Python structure (simulator-independent) when instantiating network
    simConfig.verbose = False  # show detailed messages
    simConfig.hParams['cai0_ca_ion'] = 0.0001
    simConfig.printPopAvgRates = True


    # Recording
    simConfig.recordCells = ['all']  # which cells to record from
    #simConfig.recordTraces = {'Vsoma': {'sec': 'soma', 'loc': 0.5, 'var': 'v'}
     #,'AMPA':{'sec':'dend','loc':0.5,'var':'AMPA','conds':{'cellType':'PYR'}}}
    simConfig.recordStim = True  # record spikes of cell stims
    simConfig.recordStep = 0.1  # Step size in ms to save data (eg. V traces, LFP, etc)

    external_input_populations=s_handler.stim_pop_values['external']

    x_electrodes_locations = [ [x_value[0]+1, 0, netParams.sizeZ/2]
                    for x_value in [external_input_populations[ext_pop]['x_values'][0]
                    for ext_pop in external_input_populations]]
    simConfig.recordLFP = x_electrodes_locations  # electrodes at the stim frequency

    # Saving
    simConfig.filename = fig_name  # Set file output name
    simConfig.saveFileStep = 1000  # step size in ms to save data to disk
    simConfig.savePickle = False  # True  # Whether or not to write spikes etc. to a .pkl file

    # Analysis and plotting
    simConfig.analysis['plotRaster'] = {'saveFig':
        'output_files/{}_raster.png'.format(fig_name)}  # Plot raster
    #simConfig.analysis['plotTraces'] = {'include': [5567, 5568, 5569], 'saveFig': True}  # Plot raster
    # simConfig.analysis['plot2Dnet'] = {'view': 'xz','showConns': False}  # Plot 2D net cells and connections
    #simConfig.analysis['plot2Dnet'] = {'view': 'xz',
            #'include': ['PYR23','BASK23','PYR4','BASK4'],
            #'showConns': True ,
            #'saveFig': 'output_files/{}_2Dnet.png'.format(fig_name)}  # Plot 2D net cells and connections
    simConfig.analysis['plotLFP'] = {'includeAxon': False,
         'plots': ['timeSeries'],
         'saveFig': 'output_files/{}_LFP.png'.format(fig_name)}


    return (netParams, simConfig)
