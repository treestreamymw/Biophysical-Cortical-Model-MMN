'''
This files includes various configurations
(needs to be modified to different txt files)
'''
SAMPLE_VAR = [{'gene_name':'name',
                'ref':'ref_number',
                'file':'mod_files/Ca_HVA.mod',
                'channel':'Ca_HVA',
                'variation':[{'offm': 6100, 'slom': 0.9, 'offh': -14.5,\
                        'sloh': 1.28, 'tauh': 3.52, 'c' : 0.105}]}]

SINGLE_VAR = [{'gene_name':'CACNA1C',
                'ref':'28,S26',
                'file':'mod_files/Ca_HVA.mod',
                'channel':'Ca_HVA',
                'variation':[{'offm': -31.4, 'offh': 16.3,
                    'slom':0.85, 'sloh': 0.72, 'c': 0.076}]}]


MULTI_GENE_VAR = [{'gene_name':'CACNA1C',
                'ref':'27,S24, 36,S34',
                'file':'mod_files/Ca_HVA.mod',
                'channel':'Ca_HVA',
                'variation':[ {'offm': -25.9, 'offh': -27,'c' : 0.066} ,
					{'offm': -9.8, 'slom': 0.8,'offh':-15.4,'sloh':1.05,'c': 0.181},
					{'offm': 1.3, 'offh': 1.6, 'taum': 1.45, 'tauh': 0.8, 'c':  2}  ]},

        {'gene_name':'SCN1A',
                'ref':'52,S50',
                'file':'mod_files/NaTa_t.mod',
                'channel':'NaTa_t',
                'variation':[{'offm':6, 'slom':1.16, 'tauh':1.29, 'c':0.129},
		{'offh': -26.5, 'sloh': 0.64, 'c': 0.296 },
		{'offh': 3.9, 'tauh': 0.88, 'c': 1.226}]},

        {'gene_name':'ATP2A2',
                'ref':'45,46,S43,S44',
                'file':'mod_files/CaDynamics_E2.mod',
                'channel':'CaDynamics_E2',
                'variation':[{'gamma': 0.6,
                                'c': 0.179}]}
                                ]
