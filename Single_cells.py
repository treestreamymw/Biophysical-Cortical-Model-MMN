"""
Single_cells.py

Contributors: Christoph Metzner, christoph.metzner@gmail.com, 05/08/2019
"""

import Single_cells_params
from netpyne import sim
import numpy as np


sim.createSimulateAnalyze(netParams = Single_cells_params.netParams, simConfig = Single_cells_params.simConfig)




import pylab; pylab.show()  # this line is only necessary in certain systems where figures appear empty
exit()
