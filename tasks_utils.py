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

    buffer (list) - edge 2 pyr rows and one bask row [pyr, bask], can not
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

        self.buffer = [120, 140]
        self.tasks_dict = {'oddball': partial(self.oddball_paradigm, False),
                            'flipflop': partial(self.oddball_paradigm, True),
                            'cascade': partial(self.cascade_paradigm, False),
                            'oddball_cascade': partial(self.cascade_paradigm, True),
                            'many_standards': self.many_standards_paradigm,
                            'omission': self.omission_paradigm}

        self.x_values =[]
        self.population_values=[]

    def perform_task(self):
        '''
        returns the correct function given the chosen task
        '''
        return self.tasks_dict.get(self.task, lambda: 'choose a valid task')()

    def get_pulses_range(self):
        '''
        returns the correct list of pulses Ts given the chosen task
        '''

        pulses= list(range(self.n_pulses))

        if self.task=='omission':
            for dev_ind in self.dev_indexes:
                pulses.remove(dev_ind)
            return pulses
        return pulses


    def _get_std_dev_tones(self):
        '''
        returns std and dev tones
        '''
        standard_x_values = [[self.buffer[0]-1, self.buffer[0]+1],
            [self.buffer[1]-1, self.buffer[1]+1]]

        deviant_x_values = [[self.net_x_size-self.buffer[0]-1,
            self.net_x_size-self.buffer[0]+1],
            [self.net_x_size-self.buffer[1]-1,
            self.net_x_size-self.buffer[1]+1]]

        return [standard_x_values, deviant_x_values]

    def _get_all_tones(self):
        '''
        returns all range of tones
        '''
        x_values=[]
        for t in range(self.n_pulses):
            bask_value = self.buffer[1]+ (int(t/2)*self.spacing*2)
            pyr_value = self.buffer[0]+ (t*self.spacing)

            x_values.append([[pyr_value-1,pyr_value+1],
                [bask_value-1,bask_value+1]])

            if ((pyr_value == self.net_x_size-self.buffer[0]) or
                    (bask_value == self.net_x_size-self.buffer[1])):
                break

        return x_values


    def oddball_paradigm(self, flipflop=False):
        '''
        Oddball- repetitive (standard) tone which is replaced randomly by a different (deviant) tone

        -flipflop (bool) if true presents two oddball sequences with the
                roles of deviant and standard sounds reversed

        '''
        x_values = []
        pop_values = []
        standard_x_values, deviant_x_values = self._get_std_dev_tones()

        for t in range(self.n_pulses):
            if t in self.dev_indexes:
                if not flipflop:
                    x_values.append(standard_x_values)
                    pop_values.append('std')
                elif flipflop:
                    x_values.append(deviant_x_values)
                    pop_values.append('dev')
            else:
                if not flipflop:
                    x_values.append(deviant_x_values)
                    pop_values.append('dev')
                elif flipflop:
                    x_values.append(standard_x_values)
                    pop_values.append('std')

        self.x_values = x_values
        self.population_values = pop_values

    def cascade_paradigm(self, oddball=False):
        '''
        Cascade- Tones are presented in a regular sequence of  descending
            frequencies over consecutive tones in an organized pattern.

        -oddball (bool) if True, presents a tone that breaks the pattern

        '''
        assert self.n_pulses>=4, 'can not run cascade on short simulation'
        x_values = self._get_all_tones()
        pop_values = list(range(self.n_pulses))
        if oddball:
            x_values[-1]=x_values[0]
            pop_values[-1]=pop_values[0]

        self.x_values = x_values
        self.population_values = pop_values

    def omission_paradigm(self):
        '''
        This paradigm presents a repetitive (standard)
            tone which is replaced randomly by a lack of
            stimulus with a low probability
        '''
        x_values = []
        pop_values = []
        standard_x_values, _ = self._get_std_dev_tones()

        for t in range(self.n_pulses- len(self.dev_indexes)):
            x_values.append(standard_x_values)
            pop_values.append('std')

        self.x_values = x_values
        self.population_values = pop_values


    def many_standards_paradigm(self):
        '''
        Many-standards - presents a sequence of random tones of which one
            is uniquely equal to the used deviant above
        '''
        assert self.n_pulses>=4, 'can not run cascade on short simulation'

        x_values = self._get_all_tones()
        shuffle(x_values)

        pop_values = list(range(self.n_pulses))

        self.x_values = x_values
        self.population_values = pop_values



if __name__=="__main__":
    s=Simulation_Task_Handler(560,10,40,[2,6],'omission')
    #s.oddball_paradigm(, True)
    #s.many_standards_paradigm()
    #s.cascade_paradigm(True)
    s.perform_task()
    print(s.x_values)
    print(s.get_pulses_range())
