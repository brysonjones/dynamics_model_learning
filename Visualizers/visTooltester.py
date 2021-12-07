import sys
import numpy as np
#Set up environment
sys.path.append('.')
sys.path.append('..')
from visTool import *

#Set up variables
sys.path.append("../dataLoader")
from dataLoader.DataLoader import DataLoader
DL = DataLoader('../processed_data')

DL.load_selected_data('2021-02-05-14-00-56')#"2021-02-03-16-10-37")
timeVec = DL.get_time_values()
state   = DL.get_state_data()
c       = state[:, 0:3]
c2      = c + 1

labels = ['Ref', 'Pred']

#plotState(timeVec, c, 'test', c2, labels)
#animateState(timeVec, c, 'test', c2, labels)
visualize(timeVec, c, "ShortCircles", c2, labels) #"LinearOscillations")
