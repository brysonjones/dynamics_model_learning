
import os
import sys
sys.path.append("")
sys.path.append("../model")

from model.DEL_network import *

import numpy as np
import torch.utils.data
import pandas as pd

def predict_next_state(DELN, q1, q2, u1, u2, u3):

    q_next = DELN.step(q1, q2, u1, u2, u3)


