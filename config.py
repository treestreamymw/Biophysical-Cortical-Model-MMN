
original_params={'duration':10000, 'n_pulses':10, 'n_dev':3, 'size':[1880, 2500, 1880], 'scale':1}
mini_sim_params={'duration':1000, 'n_pulses':1, 'n_dev':0, 'size':[300, 2500, 940], 'scale':12}
short_sim_params={'duration':3000, 'n_pulses':3, 'n_dev':1, 'size':[740, 2500, 940], 'scale':7}
full_sim_params={'duration':8000, 'n_pulses':8, 'n_dev':3,'size':[740, 2500, 940], 'scale':7}
short_large_sim_params={'duration':3000, 'n_pulses':3, 'n_dev':1,'size':[740, 2500, 940], 'scale':7}


SIM_PARAMS = {'full':full_sim_params,
                'short':short_sim_params,
                'orig':original_params,
                'mini': mini_sim_params,
		        'short_large': short_large_sim_params}

## origin 24*24, 12*12 -> 5760 cells
## short 8 rows, 24 cols pyr // 4 12 bask -> 480 cells
## long 14 rows, 24 cols pyr // 7 12 bask -> 840 cells



D1=0.46
D2=0.76

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
                    'gamma':'mul',
                    'decay':'mul',
                    'depth':'mul',
                    'minCai':'mul'}
