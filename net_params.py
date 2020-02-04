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

def set_params(fig_name, NET_TYPE, TASK, SEED, DEV_LIST):
    seed=SEED
    netParams=specs.NetParams()   # object of class NetParams to store the network parameters
    simConfig=specs.SimConfig()   # object of class SimConfig to store the simulation configuration

    ############################################################################
    # NETWORK PARAMETERS
    ############################################################################

    #### NETWORK PARAMETERS ####

    # Size and scale -

    ## The size of the network is defined as a 3D matrix, the scale governs the
    ## porportion between the stimuli and the network size weights
    netParams.sizeX, netParams.sizeY, netParams.sizeZ=SIM_PARAMS[NET_TYPE]['size']
    netParams.scaleConnWeight=SIM_PARAMS[NET_TYPE]['scale']
    netParams.scaleConnWeightNetStims=SIM_PARAMS[NET_TYPE]['scale']

    # Population parameters

    ## excitatory cells layer 2/3 - prediction layer
    netParams.popParams['PYR23']={'cellModel': 'PYR_Hay_asym',
                                    'cellType': 'PYR_asym',
                                    'gridSpacing': 40.0,
                                    'xRange': [20, netParams.sizeX],
                                    'yRange': [.1*netParams.sizeY,
                                                .1*netParams.sizeY],
                                    'color': 'blue'}

    ## excitatory cells layer  4 - prediction error layer
    netParams.popParams['PYR4']={'cellModel': 'PYR_Hay_four',
                                   'cellType': 'PYR_four',
                                   'gridSpacing': 40.0,
                                   'xRange': [20, netParams.sizeX],
                                   'yRange': [.2*netParams.sizeY,
                                                .2*netParams.sizeY],
                                   'color': 'green'}

    ## excitatory cells - memory layer
    netParams.popParams['PYR_memory']={'cellModel': 'PYR_Hay_mem',
                                    'cellType': 'PYR_mem',
                                    'gridSpacing': 40.0,
                                    'xRange': [20, netParams.sizeX],
                                    'yRange': [1*netParams.sizeY,
                                                1*netParams.sizeY],
                                    'color': 'purple'}

    ## inhibitory cells layer 2/3- prediction layer
    netParams.popParams['BASK23']={'cellModel': 'BASK_Vierling',
                                     'cellType': 'BASK', 'gridSpacing': 80.0,
                                     'yRange': [.1*netParams.sizeY,
                                                .1*netParams.sizeY],
                                     'color': 'red'}

    ## inhibitory cells layer 4 - prediction error layer
    netParams.popParams['BASK4']={'cellModel': 'BASK_Vierling',
                                    'cellType': 'BASK', 'gridSpacing': 80.0,
                                    'yRange': [.2*netParams.sizeY,
                                                .2*netParams.sizeY],
                                    'color': 'yellow'}


    #### CELLULAR PARAMETERS ####


    ## PYR cell of layer 4 properties - defined by the four compartment model
    cellRule=netParams.importCellParams(label='PYR4',
                                        conds={'cellType': 'PYR_four',
                                          'cellModel': 'PYR_Hay_four'},
                                        fileName='Cells/fourcompartment.hoc',
                                  		cellName='fourcompartment')

    cellRule['secs']['soma']['vinit']=-80.0
    cellRule['secs']['dend']['vinit']=-80.0
    cellRule['secs']['apic_0']['vinit']=-80.0
    cellRule['secs']['apic_1']['vinit']=-80.0

    ## PYR cell 2/3 properties - defined by the asymetrical stripped model
    cellRule=netParams.importCellParams(label='PYR23',
                                        conds={'cellType': 'PYR_asym',
                                             'cellModel': 'PYR_Hay_asym'},
                                        fileName='Cells/pyr_23_asym_stripped.hoc',
                                  		cellName='pyr_23_asym_stripped')

    ## PYR cell of memory layer - defined by the four compartment model
    cellRule=netParams.importCellParams(label='PYR_mem',
                                        conds={'cellType': 'PYR_mem',
                                            'cellModel': 'PYR_Hay_mem'},
                                        fileName='Cells/pyr_23_asym_stripped.hoc',
                                  		cellName='pyr_23_asym_stripped')





    # BASK cell properties (all layers) - defined by the fast spiking model
    cellRule=netParams.importCellParams(label='BASK', conds={'cellType': 'BASK',
                                                  'cellModel': 'BASK_Vierling'},
                                          fileName='Cells/FS.hoc',
                                          cellName='Layer2_basket')

    #### BIOPHYSICAL PARAMETERS ####

    # synapic mechanisms

    ## AMPA // GABA is defined by the Exp2Syn model
    # It is a two state kinetic scheme synapse where
    # rise time - tau1,
    # decay time- tau2
    # reversal potential - e
    # synaptic current- I (dynamic in the simulation)

    netParams.synMechParams['AMPA']={'mod': 'Exp2Syn',
                                       'tau1': 1.0,
                                       'tau2': 3.0,
                                       'e': 0.0}

    netParams.synMechParams['GABA']={'mod': 'Exp2Syn',
                                       'tau1': 5.0,
                                       'tau2': 12.0,
                                       'e': -75.0}

    #g
    # AMPASTD is defined by the FDSExp2Syn model
    # It is a two state kinetic scheme synapse where
    # rise time - tau1,
    # decay time- tau2
    # facilitation constant - f
    # facilitation time - tauF
    # reversal potential - e
    # fast depression - firing will lower the d1% of the initial
    # spike within tau_d1 ms
    # slow depression - firing will lower the d1% of the initial
    # spike within tau_d1 ms
    # synaptic current- I (dynamic in the simulation)


    netParams.synMechParams['AMPASTD']={'mod': 'FDSExp2Syn',
                                          'tau1': 1.0,
                                          'tau2': 3.0,
                                          'e': 0.0,
                                          'f': 0.0,
                                          'tau_F': 94.0,
                                          'd1': 1,
                                          'tau_D1': 380,
                                          'd2': 0.76,
                                          'tau_D2': 2500}  # only depression


    ## NMDA is defined by the NMDA model
    # It is a simple synaptic mechanism, first order kinetics of
    # binding of transmitter to postsynaptic receptors. where
    # forward (binding) rate - alpha
    # backward (unbinding) rate - beta
    # reversal potential - e
    # (max)conductance - (max)g
    # synaptic current- I (dynamic in the simulation)

    netParams.synMechParams['NMDA']={'mod': 'NMDA',
                                       'Alpha':10.0,
                                       'Beta':0.015,
                                       'e':45.0,
                                       'g':1,
                                       'gmax':1}


    ###############################################################################
    # STIMULI PARAMETERS
    ###############################################################################


    #### STIMULI GENERATION ####


    # stimuli handler given the network's params
    s_handler=Simulation_stimuli_Handler(net_x_size=netParams.sizeX,
                                n_pulses=SIM_PARAMS[NET_TYPE]['n_pulses'],
                                spacing=40.0,
                                dev_indexes=DEV_LIST,
                                task=TASK)
    s_handler.set_task_stimuli()

    # set external stimuli (i.e sensory input)
    # and internal stimuli (i.e memory trace)
    external_pulses_info=s_handler.get_formatted_pulse(external=True)
    external_pulses_time=s_handler.get_pulse_time(external=True)

    internal_pulses_info=s_handler.get_formatted_pulse(external=False)
    internal_pulses_time=s_handler.get_pulse_time(external=False)

    # set dynamic weights
    base_stimuli_weight=0.02

    weights_confidence_multiplier=[]
    added_confidence=0

    for pulse_index in range(SIM_PARAMS[NET_TYPE]['n_pulses']):
        weights_confidence_multiplier.append(1+(added_confidence*0.05))
        added_confidence=added_confidence+1
        if pulse_index==DEV_LIST[0]:
            added_confidence=0


    weights=[i*base_stimuli_weight for i in weights_confidence_multiplier]
    # generate pulses
    for t_pulse in external_pulses_info.keys():

        # External sensory stimuli
        # set stimulus name
        ext_stim_pop_name='Stim_' + \
                str(external_pulses_info[t_pulse]['pop_name']) +"_"+\
                str(t_pulse)

        # parametarize stimulus
        netParams.popParams[ext_stim_pop_name]={'cellModel': 'VecStim',
                   'numCells': 24, 'spkTimes':[0],
                   'pulses':[{'start': t_pulse*1000.0+external_pulses_time[0],
                       'end': t_pulse*1000.0+external_pulses_time[1],
                       'rate': 200,
                       'noise': 1.0}]}

        # set stimulus column
        ext_x_pyr, ext_x_bask=external_pulses_info[t_pulse]['values']

        # connect stimulus to pyramidal cells in layer 4
        netParams.connParams[ext_stim_pop_name + '->PYR4']={
            'preConds': {'popLabel': ext_stim_pop_name},
            'postConds': {'popLabel': 'PYR4', 'x': ext_x_pyr},
            'convergence': 1,
            'weight': 0.02,
            'threshold': 10,
            'synMech': 'AMPA'}

        # connect stimulus to basket cells in layer 4

        netParams.connParams[ext_stim_pop_name + '->BASK4']={
            'preConds': {'popLabel': ext_stim_pop_name},
            'postConds': {'popLabel': 'BASK4', 'x': ext_x_bask},
            'convergence': 1,
            'weight': 0.02,
            'threshold': 10,
            'synMech': 'AMPA'}

        # Internal stimuli - memory trace

        # set stimulus name
        int_stim_pop_name='internal_' + \
            str(internal_pulses_info[t_pulse]['pop_name']) +"_"+ str(t_pulse)

        # parametarize stimulus
        netParams.popParams[int_stim_pop_name]= {'cellModel': 'VecStim',
                       'numCells': 24, 'spkTimes':[0],
                       'pulses':[{'start': t_pulse*1000.0+internal_pulses_time[0],
                           'end': t_pulse*1000.0+internal_pulses_time[1],
                           'rate': 200,
                           'noise': 1.0}]}
        # set stimulus column
        int_x_pyr,int_x_bask=internal_pulses_info[t_pulse]['values']



        # connect stimulus to pyramidal cells in memory layer
        netParams.connParams[int_stim_pop_name + '->']={
            'preConds': {'popLabel': int_stim_pop_name},
            'postConds': {'popLabel': 'PYR_memory', 'x': int_x_pyr},
            'convergence': 1,
            'weight': weights[t_pulse],## increase weight of internal stim with confidence
            'threshold': 10,
            'synMech': 'AMPA'}


    ###############################################################################
    # CONNECTIVITY PARAMETERS
    ###############################################################################

    # Layer 4 intra-laminar connections
    netParams.connParams['PYR4->PYR4']={
        'preConds': {'popLabel': 'PYR4'},
        'postConds': {'popLabel': 'PYR4'},
        'sec': 'apic_1',
        'probability': '0.15*exp(-dist_3D/(4*40.0))',
        'weight': [0.0012,0.0006],
        'threshold': 10,
        'synMech': ['AMPA','NMDA']}

    netParams.connParams['PYR4->BASK4']={
        'preConds': {'popLabel': 'PYR4'},
        'postConds': {'popLabel': 'BASK4'},
        'probability': '0.45*exp(-dist_3D/(4*40.0))',
        'weight': [0.0012,0.00013],
        'threshold': 10,
        'synMech': ['AMPA','NMDA']}

    netParams.connParams['BASK4->PYR4']={
        'preConds': {'popLabel': 'BASK4'},
        'postConds': {'popLabel': 'PYR4'},
        'probability': '0.6*exp(-dist_3D/(4*40.0))',
        'weight': 0.003,
        'threshold': 10,
        'synMech': 'GABA'}

    '''
    netParams.connParams['BASK4->BASK4']={
        'preConds': {'popLabel': 'BASK4'},
        'postConds': {'popLabel': 'BASK4'},
        'probability': '0.6*exp(-dist_3D/(4*40.0))',
        'weight': 0.0,
        'threshold': 10,
        'synMech': 'GABA'}
    '''

    # Layer 2/3 intra-laminar connections
    netParams.connParams['PYR23->PYR23']={
        'preConds': {'popLabel': 'PYR23'},
        'postConds': {'popLabel': 'PYR23'},
        'sec':'oblique2b',
        'probability': '0.15*exp(-dist_3D/(1*40.0))',
        'weight':[0.0024, 0.00012],
        'threshold': 10,
        'synMech': ['AMPA','NMDA']}

    netParams.connParams['PYR23->BASK23']={
        'preConds': {'popLabel': 'PYR23'},
        'postConds': {'popLabel': 'BASK23'},
        'probability': '0.45*exp(-dist_3D/(4*40.0))',
        'weight': [0.0012,0.00013],
        'threshold': 10,
        'synMech': ['AMPA','NMDA']}


    netParams.connParams['BASK23->PYR23']={
        'preConds': {'popLabel': 'BASK23'},
        'postConds': {'popLabel': 'PYR23'},
        'sec':'oblique2a',
        'probability': '0.6*exp(-dist_3D/(2*40.0))',
        'weight': .1,
        'threshold': 10,
        'synMech': 'GABA'}


    '''netParams.connParams['BASK23->BASK23']={
        'preConds': {'popLabel': 'BASK23'},
        'postConds': {'popLabel': 'BASK23'},
        'probability': '0.6*exp(-dist_3D/(4*40.0))',
        'weight': 0.0,
        'threshold': 10,
        'synMech': 'GABA'}'''

    ## memory layer
    netParams.connParams['PYR_memory->PYR_memory']={
        'preConds': {'popLabel': 'PYR_memory'},
        'postConds': {'popLabel': 'PYR_memory'},
        'sec':'oblique2b',
        'probability': '0.15*exp(-dist_3D/(1*40.0))',
        'weight': [0.0012,0.0006],
        'threshold': 10,
        'synMech': ['AMPA','NMDA']}

    ### inter-laminar connections
    netParams.connParams['PYR4->PYR23']={
        'preConds': {'popLabel': 'PYR4'},
        'postConds': {'popLabel': 'PYR23'},
        'sec': 'basal2b',
        'probability': '.5*exp(-dist_2D/(1*40.0))', #'0.5*exp(-dist_2D/(2*40.0))',
        'weight': 0.03, #3 #0.15 #0.03 #0.3
        'threshold': 10,
        'synMech': 'AMPASTD'}


    netParams.connParams['PYR4->BASK23']={
        'preConds': {'popLabel': 'PYR4'},
        'postConds': {'popLabel': 'BASK23'},
        'probability': '0.8*exp(-dist_2D/(2*40.0))',
        'weight': 0.00015,
        'threshold': 10,
        'synMech': 'AMPASTD'}



    netParams.connParams['PYR_memory->BASK23']={
        'preConds': {'popLabel': 'PYR_memory'},
        'postConds': {'popLabel': 'BASK23'},
        'probability': '0.8*exp(-dist_3D/(2*40.0))', # 0.8*exp(-dist_3D/(3*40.0))...
        'weight': 0.001,#0.015,#0.00015,
        'threshold': 10,
        'synMech': 'AMPASTD'}


    ########################################################################
    # SIMULATION PARAMETERS
    ########################################################################

    # Simulation parameters
    simConfig.hParams['celsius']=30.0
    simConfig.duration=SIM_PARAMS[NET_TYPE]['duration']  # Duration of the simulation, in ms
    simConfig.dt=0.05  # Internal integration timestep to use
    simConfig.seeds={'conn': seed, 'stim': seed, 'loc': seed}  # Seeds for randomizers (conn., input stim. and cell loc.)
    simConfig.createNEURONObj=1  # create HOC objects when instantiating network
    simConfig.createPyStruct=1  # create Python structure (simulator-independent) when instantiating network
    simConfig.verbose=False  # show detailed messages
    simConfig.hParams['cai0_ca_ion']=0.0001
    simConfig.printPopAvgRates=True


    # Recording
    simConfig.recordCells=['all']  # which cells to record from

    #recod membrane potential
    '''
    simConfig.recordTraces={}

    simConfig.recordTraces['BASK23'] = {'sec':'soma','loc':0.5,'var':'v',
            'conds':{'pop':'BASK23', 'cellType':'BASK'}}

    simConfig.recordTraces['PYR4_PYR23'] =\
            {'Vsoma':{'sec':'soma','loc':0.5,'var':'v'},
            'PYR4-PYR23':{'sec':'basal2b','loc':0.5,
                'var':'AMPASTD','conds':{'cellType':'PYR23'}}}

    simConfig.recordTraces['PYR23_PYR23_AMPA'] =\
            {'Vsoma':{'sec':'soma','loc':0.5,'var':'v'},
            'PYR23-PYR23':{'sec':'oblique2b','loc':0.5,
                'var':'AMPA','conds':{'cellType':'PYR23'}}}

    simConfig.recordTraces['PYR23_PYR23_NMDA'] =\
            {'Vsoma':{'sec':'soma','loc':0.5,'var':'v'},
            'PYR23-PYR23':{'sec':'oblique2b','loc':0.5,
                'var':'NMDA','conds':{'cellType':'PYR23'}}}

    '''
    simConfig.recordStim=True  # record spikes of cell stims
    simConfig.recordStep=0.1  # Step size in ms to save data (eg. V traces, LFP, etc)

    external_input_populations=s_handler.stim_pop_values['external']

    x_electrodes_locations=[ [x_value[0]+1, 0, \
                        netParams.sizeZ/2] for x_value in \
                    [external_input_populations[ext_pop]['x_values'][0]
                    for ext_pop in external_input_populations] ]

     # electrodes at the stim frequency
    simConfig.recordLFP=x_electrodes_locations
    # Saving
    simConfig.saveFolder='output_files/'
    simConfig.filename='output_files/{}'.format(fig_name)  # Set file output name
    simConfig.saveFileStep=1000  # step size in ms to save data to disk
    simConfig.savePickle=False
    simConfig.saveJson=True

    # Analysis and plotting
    simConfig.analysis['plotRaster']={'saveFig':
        'output_files/{}_raster.png'.format(fig_name)}

    #simConfig.analysis['plotSpikeHist']={
    #        'include': ['PYR23','PYR_memory'],
    #        'yaxis':'count',
    #        'graphType':'line',
    #        'saveFig': 'output_files/{}_plotSpikeHist.png'.format(fig_name)}

    simConfig.analysis['plotLFP']={'includeAxon': False,
         'plots': ['timeSeries'],
         'saveFig': 'output_files/{}_LFP.png'.format(fig_name)}

    simConfig.analysis['plotConn']={'include':['all'], 'feature':'strength',
        'orderBy':'gid', 'figSize':(10,10), 'groupBy':'pop',
         'saveFig': 'output_files/{}_conn.png'.format(fig_name)}


    #Plot 2D net cells and connections
    #simConfig.analysis['plot2Dnet']={'view': 'xy',
    #        'include': ['PYR23', 'PYR4', 'BASK23', 'BASK4', 'PYR_memory'],
    #        'showConns': True ,
    #        'saveFig': 'output_files/{}_2Dnet.png'.format(fig_name)}


    return (netParams, simConfig)
