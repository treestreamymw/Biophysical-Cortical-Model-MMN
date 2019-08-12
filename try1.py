from net_params import set_params
from netpyne import sim
import numpy as np

print ('d')
NP, SC = set_params(fig_name='try,png', NET_TYPE='short', TASK='f')
sim.createSimulateAnalyze(netParams=NP, simConfig=SC)
