import numpy as np
from config import SIM_PARAMS
from random import shuffle
from functools import partial

class Simulation_stimuli_Handler(object):
    """
    This class handles the specific stimuli for the simulation dependent on the
        specific task

    params:
    net_x_size (int) - the size of the network on the x axis,
        indicates the number of rows
    n_pulses (int) - the numebr of pulses in the simulation
    spacing (int) the interneuron spacing
    dev_indexes (int) - the indexes of the deviant stimulus
    task (str) - the task required
    buffer (list) - Marks the edge rows [pyr, bask], which can not
        be used in stimulus
    task_dicts (dict) - maps the user input to the relevant task

    external_population_values (list)-
    internal_population_values
    """

    def __init__(self, net_x_size, n_pulses, spacing, dev_indexes ,task):
        self.net_x_size=net_x_size
        self.n_pulses=n_pulses
        self.spacing=spacing
        self.dev_indexes=dev_indexes
        self.task=task
        self.buffer={'pyr':20+self.spacing,'bask':2*self.spacing} #{'pyr':60,'bask':80}
        self.tasks_dict={'oddball': partial(self.oddball_paradigm, False, False),
                            'flipflop': partial(self.oddball_paradigm, True, False),
                            'cascade': partial(self.cascade_paradigm, False),
                            'oddball_cascade': partial(self.cascade_paradigm, True),
                            'many_standards': self.many_standards_paradigm,
                            'no_oddball':self.no_oddball_paradigm,
                            'omission': partial(self.oddball_paradigm, False, True)}

        self.stim_pop_values={'external':[], 'internal':[]}

        self.stimulus_time={'external':[500,700], 'internal':[500,700]}

    def set_task_stimuli(self):
        '''
        Returns the correct function given the chosen task
        '''
        return self.tasks_dict.get(self.task, lambda: 'choose a valid task')()

    def _get_all_available_tones(self):
        '''
        returns a list of all vailable tones given the cortical layout
        '''
        x_values=[]
        for t in range(self.n_pulses):

            bask_value = self.buffer['bask'] + (t*self.spacing*2)
            pyr_value = self.buffer['pyr'] + (t*self.spacing*2)

            x_values.append([[pyr_value-1,pyr_value+1],
                [bask_value-1,bask_value+1]])

            if ((pyr_value == self.net_x_size-self.buffer['pyr']) or
                    (bask_value == self.net_x_size-self.buffer['bask'])):
                break

        return x_values

    def _get_std_dev_tones(self):
            '''
            returns the first and last available tones which represnt the
            furthest possible tones within the network
            '''
            all_tone=self._get_all_available_tones()
            return [all_tone[0],all_tone[-1]]

    def no_oddball_paradigm(self):
        '''
        normal stimuli no oddball

        '''
        # stores the x values and indexes for the different stimuli population
        # internally and externally

        internal_pop_values = {}
        external_pop_values = {}

        deviant_x_values, standard_x_values = self._get_all_available_tones()[2]

        pop_values = {i:{'x_values': deviant_x_values,
        'pulses':[i]} for i in range(self.n_pulses)}

        # generally both take the std cascade
        self.stim_pop_values['internal'] = pop_values
        self.stim_pop_values['external'] = pop_values


    def oddball_paradigm(self, flipflop=False, omission=False):
        '''
        Oddball- repetitive (standard) tone which is replaced randomly by
            a different (deviant) tone

        -flipflop (bool) if true presents two oddball sequences with the
                roles of deviant and standard sounds reversed
        - omission (bool) if true - removes the dev stimulus, if false - creates
            a dev tone

        '''
        # stores the x values and indexes for the different stimuli population
        # internally and externally

        internal_pop_values = {}
        external_pop_values = {}

        if flipflop:
            deviant_x_values, standard_x_values = self._get_std_dev_tones()
        else:
            standard_x_values, deviant_x_values = self._get_std_dev_tones()
        # external includes both std and dev based on the dev indexes
        if omission:
            external_pop_values['dev']={'x_values':[[0,0],[0,0]],
                            'pulses':self.dev_indexes}
        if not omission:
            external_pop_values['dev']={'x_values':deviant_x_values,
                            'pulses':self.dev_indexes}

        external_pop_values['std']={'x_values':standard_x_values,
                            'pulses':list(set(range(self.n_pulses)).\
                                    difference(self.dev_indexes))}

        #internal includes only std for all pulses indexes
        internal_pop_values['std']= {'x_values':standard_x_values,
                            'pulses':list(range(self.n_pulses))}
        internal_pop_values['dev']={}


        self.stim_pop_values['external']=external_pop_values
        self.stim_pop_values['internal']=internal_pop_values

    def cascade_paradigm(self, oddball=False):
        '''
        Cascade- Tones are presented in a regular sequence of  descending
            frequencies over consecutive tones in an organized pattern.

        -oddball (bool) if True, presents a tone that breaks the pattern
            (first returns as last)

        '''
        x_values = self._get_all_available_tones()
        int_pop_values = {i:{'x_values': x_values[i],
        'pulses':[i]} for i in range(self.n_pulses)}
        ext_pop_values = {i:{'x_values': x_values[i],
        'pulses':[i]} for i in range(self.n_pulses)}

        # generally both take the std cascade
        self.stim_pop_values['internal'] = int_pop_values
        self.stim_pop_values['external'] = ext_pop_values
        # if odd ball, the external popluation introduces a dev tone

        if oddball:
            self.stim_pop_values['external'][self.
                dev_indexes[0]]['x_values']= ext_pop_values[2]['x_values']


    def many_standards_paradigm(self):
        '''
        Many-standards - presents a sequence of random tones of which one
            is uniquely equal to the used deviant above

            it is random so there is no prediction
        '''

        x_values = self._get_all_available_tones()
        ext_pop_values = {i:{'x_values': x_values[i],
        'pulses':[j]} for i,j in zip(range(self.n_pulses),[6,4,7,5,1,0,3,2])}


        self.stim_pop_values['external']=ext_pop_values
        self.stim_pop_values['internal']={i:{'x_values': x_values[i],
        'pulses':[j]} for i,j in zip(range(self.n_pulses),[3,2,7,1,6,4,0,5])}

    def get_formatted_pulse(self, external=True):
        '''
        returns the pulses in the format needed in neuron based on the parameters
        defined by the task specification
        '''
        # the
        pulse_dict = {i:{'pop_name':' ', 'values':[]} for i in range(self.n_pulses)}

        if external:
            stimuli_origin='external'
        else:
            stimuli_origin='internal'
        for pop in self.stim_pop_values[stimuli_origin]:

            if self.stim_pop_values[stimuli_origin][pop] != {}:
                for pulse in self.stim_pop_values[stimuli_origin][pop]['pulses']:
                    pulse_dict[pulse]['pop_name']=pop
                    pulse_dict[pulse]['values']=\
                            self.stim_pop_values[stimuli_origin][pop]['x_values']

        return pulse_dict

    def get_pulse_time(self, external=True):
        if external:
            return self.stimulus_time['external']
        else:
            return self.stimulus_time['internal']

if __name__=="__main__":
    TASK='no_oddball'
    NET_TYPE='full'
    s=Simulation_stimuli_Handler(300 ,8 ,40,[6],TASK)

    s.set_task_stimuli()


    pulses_info=s.get_formatted_pulse(external=True)
    pulses_info_in=s.get_formatted_pulse(external=False)

    times=s.get_pulse_time(external=True)
    print(pulses_info)
    #pulses_info_ext=s.get_formatted_pulse(external=True)


    netparams={}

    for t_pulse in pulses_info.keys():
        stim='Stim_' + str(pulses_info[t_pulse]['pop_name']) + str(t_pulse)

        netparams[stim] = {'cellModel': 'VecStim',
                   'numCells': 24, 'spkTimes':[0],
                   'pulses':[{'start': t_pulse*1000+times[0],
                       'end': t_pulse*1000.0+times[1], 'rate': 200, 'noise': 1.0}]}
        #print(pulses_info[t_pulse]['values'])
        #print(pulses_info_in[t_pulse]['values'])
        x_pyr, x_bask=pulses_info[t_pulse]['values']
        _, _=pulses_info_in[t_pulse]['values']


    print (netparams)
