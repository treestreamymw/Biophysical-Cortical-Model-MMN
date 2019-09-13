
original_params={'duration':10000, 'n_pulses':10, 'n_dev':3, 'size':[1880, 2500, 1880], 'scale':1}

test_params={'duration':1500, 'n_pulses':2, 'n_dev':0, 'size':[360, 2500, 920], 'scale':10}

short_sim_params={'duration':3000, 'n_pulses':3, 'n_dev':1, 'size':[360, 2500, 920], 'scale':10}
full_sim_params={'duration':10000, 'n_pulses':10, 'n_dev':3,'size':[520, 2500, 920], 'scale':6}

SIM_PARAMS = {'full':full_sim_params,
                'short':short_sim_params,
                'orig':original_params,
                'test':test_params}



D1 = 0.46
D2 = 0.76

LOW_GABA=0.002
HIGH_GABA=0.003

JSONS_DIR = './jsons'

N_PEAKS = 10
N_MSRMNT = 10000


VAR_FACTOR_TYPE = {'offm':'add',
                    'offh':'add',
                    'slom':'mul',
                    'sloh':'mul',
                    'taum':'mul',
                    'tauh':'mul',
                    'gamma':'mul'}
