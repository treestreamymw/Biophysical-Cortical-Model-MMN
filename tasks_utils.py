import numpy as np
from config import SIM_PARAMS
from random import shuffle
from functools import partial

class Simulation_Task_Handler(object):
    """
    handles the specific task for the simulation

    net_x_size (int) - the size of the network on the x axis,
        indicates the number of rows
    n_pulses (int) - the numebr of stimulus pulses
    task (str) - the task required

    buffer (list) - marks the edge  pyr rows and one bask row [pyr, bask], can not
        be used in stimulus
    task_dicts (dict) -gets user info and calls the relevant tasks
    x_values (list of tupples)- filled by the task function,
        represents the x values for the simulation. (i.e the x position for
        the basket and pyramidal cells)
    population_values (list)- filled with the name of the populations
        std/dev for oddball and index for cascade.
    """

    def __init__(self, net_x_size, n_pulses, spacing, dev_indexes ,task):
        self.net_x_size = net_x_size
        self.n_pulses = n_pulses
        self.spacing = spacing
        self.dev_indexes=dev_indexes
        self.task = task

        self.buffer = {'pyr':60,'bask':80}
        self.tasks_dict = {'oddball': partial(self.oddball_paradigm, False),
                            'flipflop': partial(self.oddball_paradigm, True),
                            'cascade': partial(self.cascade_paradigm, False),
                            'oddball_cascade': partial(self.cascade_paradigm, True),
                            'many_standards': self.many_standards_paradigm,
                            'omission': self.omission_paradigm}

        #self.x_values =[]
        self.population_values=[]

    def set_task_stimuli(self):
        '''
        returns the correct function given the chosen task
        '''
        return self.tasks_dict.get(self.task, lambda: 'choose a valid task')()

    '''def get_pulses_range(self):

        #returns the correct list of pulses Ts given the chosen task


        pulses= list(range(self.n_pulses))

        if self.task=='omission':
            for dev_ind in self.dev_indexes:
                pulses.remove(dev_ind)
            return pulses
        return pulses'''

    def _get_all_tones(self):
        '''
        returns all range of tones
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
            returns std and dev tones
            '''
            all_tone=self._get_all_tones()
            return [all_tone[0],all_tone[-1]]

    def oddball_paradigm(self, flipflop=False):
        '''
        Oddball- repetitive (standard) tone which is replaced randomly by a different (deviant) tone

        -flipflop (bool) if true presents two oddball sequences with the
                roles of deviant and standard sounds reversed

        '''
        pop_values = {}
        standard_x_values, deviant_x_values = self._get_std_dev_tones()

        pop_values['dev']= {'x_values':deviant_x_values,
                            'pulses':self.dev_indexes}
        pop_values['std']= {'x_values':standard_x_values,
                            'pulses':set(range(self.n_pulses)).difference(self.dev_indexes)}

        self.population_values = pop_values

    def cascade_paradigm(self, oddball=False):
        '''
        Cascade- Tones are presented in a regular sequence of  descending
            frequencies over consecutive tones in an organized pattern.

        -oddball (bool) if True, presents a tone that breaks the pattern

        '''
        assert self.n_pulses>=4, 'can not run cascade on short simulation'
        x_values = self._get_all_tones()
        pop_values = {i:{'x_values': x_values[i], 'pulses':[i]} for i in range(self.n_pulses)}

        if oddball:
            pop_values[0]['pulses'].append(self.n_pulses-1)
            del pop_values[self.n_pulses-1]

        self.population_values = pop_values

    def omission_paradigm(self):
        '''
        This paradigm presents a repetitive (standard)
            tone which is replaced randomly by a lack of
            stimulus with a low probability
        '''
        standard_x_values, _ = self._get_std_dev_tones()
        pop_values = {'std':{'x_values':standard_x_values,
            'pulses':list(set(range(self.n_pulses)).difference(self.dev_indexes))}}


        self.population_values = pop_values

    def many_standards_paradigm(self):
        '''
        Many-standards - presents a sequence of random tones of which one
            is uniquely equal to the used deviant above
        '''
        assert self.n_pulses>=4, 'can not run cascade on short simulation'

        x_values = self._get_all_tones()
        shuffle(x_values)
        pop_values = {i:{'x_values': x_values[i], 'pulses':[i]} for i in range(self.n_pulses)}


        self.population_values = pop_values

    def get_details_in_pulses(self):
        x_values = {i:{'pop_name':'stim', 'values':[]} for i in range(self.n_pulses)}
        for pop in self.population_values:
            for pulse in self.population_values[pop]['pulses']:
                x_values[pulse]['pop_name']=pop
                x_values[pulse]['values']=self.population_values[pop]['x_values']

        return x_values



if __name__=="__main__":
    s=Simulation_Task_Handler(300 ,3 ,40,[2],'oddball')
    #s.oddball_paradigm(, True)
    #s.many_standards_paradigm()
    #s.cascade_paradigm(True)
    s.set_task_stimuli()
    #print(s.population_values)
    #print([[g[0]+1,0,8] for g in [s.population_values[i]['x_values'][0] for i in s.population_values]])
    NET_TYPE='short'
    TASK='oddball'
    pulses_info=s.get_details_in_pulses()

    deviant_pulses_indexes = np.random.choice(list(range(3)),
                SIM_PARAMS['short']['n_dev'], replace=False)
    print(deviant_pulses_indexes)
    print(pulses_info.keys())


    stimuli_pulses = [{'start': t_pulse*1000+500.0,
        'end': t_pulse*1000.0+700.0, 'rate': 200, 'noise': 1.0,
        'spkTimes':[1]}
         for t_pulse in pulses_info.keys()]

    netparams={}

    for t_pulse in pulses_info.keys():

        stim='Stim_' + pulses_info[t_pulse]['pop_name']

        if stim in netparams.keys():
            netparams[stim]['pulses'].append(stimuli_pulses[t_pulse])
        else:
            netparams[stim] = {'cellModel': 'VecStim',
                   'numCells': 24, 'pulses':[stimuli_pulses[t_pulse]]}

        x_pyr, x_bask=pulses_info[t_pulse]['values']
        print( x_pyr, x_bask)
    print (netparams)
