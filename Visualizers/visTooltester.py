import sys
import numpy as np
#Set up environment
sys.path.append('.')
from visTool import plotState
from visTool import animateState
from visTool import visualize

#Set up variables
zline = np.linspace(0,15,1000)
xline = np.sin(zline)
yline = np.cos(zline)

c = [zline, xline, yline]

timeVec = zline

#plotState(timeVec, c)
#animateState(timeVec, c)
visualize(timeVec, c)
