import sys
import numpy as np
#Set up environment
sys.path.append('.')
sys.path.append('..')
from visTool import plotState
from visTool import animateState
from visTool import visualize

#Set up variables
sys.path.append("C:/Users/zvick/OneDrive/Documents/GitHub/dynamics_model_learning/dataLoader")
from dataLoader.DataLoader import DataLoader
DL = DataLoader("C:/Users/zvick/OneDrive/Documents/GitHub/dynamics_model_learning/processed_data")

DL.load_selected_data('2021-02-05-14-00-56')#"2021-02-03-16-10-37")
timeVec = DL.get_time_values()
state   = DL.get_state_data()
c       = state[:, 16:19]

#plotState(timeVec, c)
#animateState(timeVec, c)
visualize(timeVec, c, "ShortCircles")#"LinearOscillations")
