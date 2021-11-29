
import os
import sys
sys.path.append("")
sys.path.append("../model")

from model.DEL_network import *

import numpy as np
import torch.utils.data
import pandas as pd

def simulate_(DELN, q1, q2, u_array):
    N = u_array.shape[0]
    qhist = np.zeros((N, 7))
    qhist[0, :] = q1
    qhist[1, :] = q2

    for i in range(2, N):
        qhist[:, i] = DELN.step(qhist[i-2:i, :], u_array[i-2:i+1, :])
